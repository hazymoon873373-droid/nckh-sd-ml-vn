#!/usr/bin/env python3
"""
engine.py — Real-data Quantile Regression Forests (Meinshausen, 2006)
+ Almost Stochastic Dominance (Leshno & Levy, 2002; Tzeng et al., 2013)
+ Sector Capital Flow & Out-of-Sample Walk-Forward Backtest for Vietnamese HOSE stocks.
"""

from __future__ import annotations

import datetime as dt
import json
import pathlib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent
CACHE_PATH = BASE_DIR / "vn_market_cache.json"
MODEL_OUTPUT_PATH = BASE_DIR / "model_results.json"

STOCK_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "VCB.VN", "name": "Vietcombank", "sector": "Ngân hàng"},
    {"ticker": "TCB.VN", "name": "Techcombank", "sector": "Ngân hàng"},
    {"ticker": "MBB.VN", "name": "MB Bank", "sector": "Ngân hàng"},
    {"ticker": "ACB.VN", "name": "ACB", "sector": "Ngân hàng"},
    {"ticker": "BID.VN", "name": "BIDV", "sector": "Ngân hàng"},
    {"ticker": "CTG.VN", "name": "VietinBank", "sector": "Ngân hàng"},
    {"ticker": "FPT.VN", "name": "FPT Corp", "sector": "Công nghệ & Bán lẻ"},
    {"ticker": "MWG.VN", "name": "Thế Giới Di Động", "sector": "Công nghệ & Bán lẻ"},
    {"ticker": "FRT.VN", "name": "FPT Retail", "sector": "Công nghệ & Bán lẻ"},
    {"ticker": "PNJ.VN", "name": "Vàng bạc Phú Nhuận", "sector": "Công nghệ & Bán lẻ"},
    {"ticker": "HPG.VN", "name": "Hòa Phát", "sector": "Thép & Hóa chất"},
    {"ticker": "DGC.VN", "name": "Hóa chất Đức Giang", "sector": "Thép & Hóa chất"},
    {"ticker": "SSI.VN", "name": "Chứng khoán SSI", "sector": "Chứng khoán"},
    {"ticker": "VCI.VN", "name": "Chứng khoán Vietcap", "sector": "Chứng khoán"},
    {"ticker": "VHM.VN", "name": "Vinhomes", "sector": "Bất động sản"},
    {"ticker": "KDH.VN", "name": "Nhà Khang Điền", "sector": "Bất động sản"},
    {"ticker": "NLG.VN", "name": "Nam Long Group", "sector": "Bất động sản"},
    {"ticker": "VNM.VN", "name": "Vinamilk", "sector": "Tiêu dùng & Năng lượng"},
    {"ticker": "MSN.VN", "name": "Masan Group", "sector": "Tiêu dùng & Năng lượng"},
    {"ticker": "GAS.VN", "name": "PV Gas", "sector": "Tiêu dùng & Năng lượng"},
    {"ticker": "PVD.VN", "name": "PV Drilling", "sector": "Tiêu dùng & Năng lượng"},
]

HORIZON_DAYS = 22  # ~1 trading month forward
CDF_GRID = np.linspace(-0.28, 0.36, 129)  # shared return grid for FSD/SSD/ASD integration
QUANTILE_PROBS = np.linspace(0.05, 0.95, 19)
TRANSACTION_COST_ROUNDTRIP = 0.0035  # 0.25% brokerage (2 sides) + 0.10% selling tax in VN


def fetch_yahoo_vn_data(force_refresh: bool = False) -> dict[str, Any]:
    """Fetch 3y daily OHLCV from Yahoo Finance v8 chart API, with local JSON cache."""
    if CACHE_PATH.exists() and not force_refresh:
        return json.loads(CACHE_PATH.read_text(encoding="utf-8"))

    raw_series: dict[str, Any] = {}
    for meta in STOCK_UNIVERSE:
        t = meta["ticker"]
        url = (
            f"https://query1.finance.yahoo.com/v8/finance/chart/"
            f"{urllib.parse.quote(t)}?range=3y&interval=1d&events=div,splits"
        )
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        res = payload["chart"]["result"][0]
        ts = res["timestamp"]
        q = res["indicators"]["quote"][0]
        adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])

        rows = []
        for idx, unix_t in enumerate(ts):
            c = adj[idx] if adj[idx] is not None else q["close"][idx]
            raw_c = q["close"][idx]
            h = q["high"][idx]
            l = q["low"][idx]
            o = q["open"][idx]
            v = q["volume"][idx]
            if c is None or raw_c is None or h is None or l is None or v is None:
                continue
            date_str = dt.datetime.fromtimestamp(unix_t, tz=dt.timezone.utc).strftime("%Y-%m-%d")
            rows.append(
                {
                    "date": date_str,
                    "open": float(o) if o is not None else float(raw_c),
                    "high": float(h),
                    "low": float(l),
                    "close_raw": float(raw_c),
                    "adj_close": float(c),
                    "volume": float(v),
                }
            )
        raw_series[t] = {
            "ticker": t,
            "name": meta["name"],
            "sector": meta["sector"],
            "rows": rows,
        }

    bundle = {
        "fetched_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": "Yahoo Finance v8 Chart API (.VN daily OHLCV, 3y)",
        "stocks": raw_series,
    }
    CACHE_PATH.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    return bundle


def sanitize_and_build_panel(bundle: dict[str, Any]) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """
    Align dates across all 21 HOSE tickers, detect/repair Yahoo Finance .VN
    unadjusted corporate-action jumps (|daily return| > 7.5% on HOSE), and compute features.
    """
    audit_events: list[dict[str, Any]] = []
    frames: list[pd.DataFrame] = []

    for ticker, info in bundle["stocks"].items():
        df = pd.DataFrame(info["rows"]).drop_duplicates(subset=["date"]).sort_values("date").reset_index(drop=True)
        # Recompute clean return series where single-day jump > 7.5% is a Yahoo split-date offset
        px = df["adj_close"].to_numpy(dtype=float)
        clean_ret = np.zeros(len(px), dtype=float)
        for i in range(1, len(px)):
            r = (px[i] / px[i - 1]) - 1.0 if px[i - 1] > 0 else 0.0
            if abs(r) > 0.075:
                audit_events.append(
                    {
                        "ticker": ticker,
                        "date": df.loc[i, "date"],
                        "raw_daily_ret_pct": round(r * 100.0, 2),
                        "action": "Capped/neutralized corporate-action jump outside HOSE +/-7% band",
                    }
                )
                r = 0.0
            clean_ret[i] = r

        # Reconstruct split-clean synthetic index price starting at 100
        clean_px = 100.0 * np.cumprod(1.0 + clean_ret)
        df["ticker"] = ticker
        df["name"] = info["name"]
        df["sector"] = info["sector"]
        df["ret_1d"] = clean_ret
        df["clean_px"] = clean_px
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)

    # Equal-weighted 21-stock HOSE Leaders Market Benchmark
    bench_daily = panel.groupby("date", as_index=False)["ret_1d"].mean().rename(columns={"ret_1d": "bench_ret_1d"})
    bench_daily = bench_daily.sort_values("date").reset_index(drop=True)
    bench_daily["bench_px"] = 100.0 * np.cumprod(1.0 + bench_daily["bench_ret_1d"].to_numpy())
    panel = panel.merge(bench_daily[["date", "bench_ret_1d", "bench_px"]], on="date", how="left")

    # Compute features per ticker (strictly causal: uses data up to t only)
    enriched: list[pd.DataFrame] = []
    for ticker, grp in panel.groupby("ticker", sort=False):
        g = grp.sort_values("date").reset_index(drop=True).copy()
        r = g["ret_1d"]
        p = g["clean_px"]
        bp = g["bench_px"]

        g["ret_5d"] = p / p.shift(5) - 1.0
        g["ret_22d"] = p / p.shift(22) - 1.0
        g["ret_63d"] = p / p.shift(63) - 1.0
        g["rel_strength_22d"] = g["ret_22d"] - (bp / bp.shift(22) - 1.0)
        # EWMA volatility (RiskMetrics lambda = 0.94, Barunik et al. 2026)
        g["ewma_vol"] = np.sqrt((r ** 2).ewm(alpha=0.06, adjust=False).mean()) * np.sqrt(252.0)
        g["skew_22d"] = r.rolling(22).skew().fillna(0.0)
        g["max_ret_22d"] = r.rolling(22).max()  # Fong (2013) lottery proxy

        # Money flow indicators from OHLCV (CMF-20, Abnormal Turnover, MFI-14)
        hl_span = (g["high"] - g["low"]).replace(0.0, np.nan)
        mfm = ((g["close_raw"] - g["low"]) - (g["high"] - g["close_raw"])) / hl_span
        mfm = mfm.fillna(0.0).clip(-1.0, 1.0)
        mfv = mfm * g["volume"]
        vol_sum_20 = g["volume"].rolling(20).sum().replace(0.0, np.nan)
        g["cmf_20"] = (mfv.rolling(20).sum() / vol_sum_20).fillna(0.0)

        dollar_vol = g["close_raw"] * g["volume"]
        base_dvol = dollar_vol.rolling(20).mean().shift(1).replace(0.0, np.nan)
        g["ab_turn"] = (dollar_vol.rolling(5).mean() / base_dvol).clip(0.2, 5.0).fillna(1.0)
        g["flow_score"] = g["cmf_20"] * g["ab_turn"]

        # Forward 22-day return target (strictly future: t -> t+22)
        g["fwd_ret_22d"] = p.shift(-HORIZON_DAYS) / p - 1.0
        g["fwd_bench_22d"] = bp.shift(-HORIZON_DAYS) / bp - 1.0
        g["fwd_cmf_20"] = g["cmf_20"].shift(-HORIZON_DAYS)
        g["fwd_ab_turn"] = g["ab_turn"].shift(-HORIZON_DAYS)
        enriched.append(g)

    full = pd.concat(enriched, ignore_index=True)
    return full, audit_events


@dataclass
class SimpleTree:
    feature_idx: int
    threshold: float
    left: Any
    right: Any
    leaf_indices: np.ndarray | None = None


class QuantileRandomForest:
    """
    Pure-NumPy implementation of Meinshausen (2006, JMLR) Quantile Regression Forests.
    Produces full monotonic conditional CDFs F(y | X=x) via adaptive leaf sample weights.
    """

    def __init__(self, n_trees: int = 50, max_depth: int = 4, min_samples_leaf: int = 25, seed: int = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rng = np.random.default_rng(seed)
        self.trees: list[SimpleTree] = []
        self.y_train: np.ndarray = np.array([])

    def _build_tree(self, X: np.ndarray, y: np.ndarray, idx: np.ndarray, depth: int) -> SimpleTree:
        if depth >= self.max_depth or len(idx) <= 2 * self.min_samples_leaf:
            return SimpleTree(-1, 0.0, None, None, leaf_indices=idx)

        n_feats = X.shape[1]
        feat_candidates = self.rng.choice(n_feats, size=max(2, int(np.sqrt(n_feats)) + 1), replace=False)
        best_score = np.inf
        best_feat = -1
        best_thr = 0.0
        best_left = None
        best_right = None

        y_sub = y[idx]
        for f in feat_candidates:
            vals = X[idx, f]
            qs = np.quantile(vals, [0.25, 0.5, 0.75])
            for thr in np.unique(qs):
                mask_l = vals <= thr
                nl = int(mask_l.sum())
                nr = len(idx) - nl
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                yl = y_sub[mask_l]
                yr = y_sub[~mask_l]
                score = nl * np.var(yl) + nr * np.var(yr)
                if score < best_score:
                    best_score = score
                    best_feat = int(f)
                    best_thr = float(thr)
                    best_left = idx[mask_l]
                    best_right = idx[~mask_l]

        if best_feat == -1 or best_left is None or best_right is None:
            return SimpleTree(-1, 0.0, None, None, leaf_indices=idx)

        left_node = self._build_tree(X, y, best_left, depth + 1)
        right_node = self._build_tree(X, y, best_right, depth + 1)
        return SimpleTree(best_feat, best_thr, left_node, right_node, leaf_indices=None)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.y_train = y
        self.trees = []
        n = len(y)
        for _ in range(self.n_trees):
            boot = self.rng.integers(0, n, size=n)
            self.trees.append(self._build_tree(X, y, boot, 0))

    def _leaf_for_row(self, node: SimpleTree, x: np.ndarray) -> np.ndarray:
        while node.leaf_indices is None:
            node = node.left if x[node.feature_idx] <= node.threshold else node.right
        return node.leaf_indices

    def predict_cdfs(self, X_query: np.ndarray, grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (cdfs [M, G], means [M], quantiles [M, Q]) using Meinshausen Eq. (6)."""
        m = X_query.shape[0]
        cdfs = np.zeros((m, len(grid)), dtype=float)
        means = np.zeros(m, dtype=float)
        quants = np.zeros((m, len(QUANTILE_PROBS)), dtype=float)

        for i in range(m):
            x = X_query[i]
            weights = np.zeros(len(self.y_train), dtype=float)
            for tree in self.trees:
                leaf = self._leaf_for_row(tree, x)
                w = 1.0 / (self.n_trees * len(leaf))
                np.add.at(weights, leaf, w)

            order = np.argsort(self.y_train)
            y_sorted = self.y_train[order]
            w_sorted = weights[order]
            cum_w = np.cumsum(w_sorted)
            cum_w = cum_w / cum_w[-1]

            cdfs[i] = np.interp(grid, y_sorted, cum_w, left=0.0, right=1.0)
            means[i] = float(np.sum(weights * self.y_train))
            quants[i] = np.interp(QUANTILE_PROBS, cum_w, y_sorted)

        return cdfs, means, quants


def compute_sd_metrics(
    cdf_a: np.ndarray,
    cdf_b: np.ndarray,
    mean_a: float,
    mean_b: float,
    grid: np.ndarray,
) -> dict[str, Any]:
    """
    Exact numerical computation of FSD, SSD, AFSD (Leshno & Levy 2002),
    and ASSD (Tzeng, Huang & Shih 2013) on a shared return grid.
    """
    dx = float(grid[1] - grid[0])

    # First-order dominance violation (A > B means F_A(x) <= F_B(x))
    diff1 = cdf_a - cdf_b
    pos1 = np.maximum(diff1, 0.0)
    abs1 = np.abs(diff1)
    denom1 = float(np.trapezoid(abs1, dx=dx))
    eps1_star = float(np.trapezoid(pos1, dx=dx) / denom1) if denom1 > 1e-12 else 0.0
    fsd_strict = bool(np.all(diff1 <= 1e-9) and np.any(diff1 < -1e-6))

    # Integrated CDFs F^(2)(t) = int_a^t F(x) dx
    f2_a = np.zeros_like(cdf_a)
    f2_b = np.zeros_like(cdf_b)
    f2_a[1:] = np.cumsum(0.5 * (cdf_a[:-1] + cdf_a[1:]) * dx)
    f2_b[1:] = np.cumsum(0.5 * (cdf_b[:-1] + cdf_b[1:]) * dx)

    diff2 = f2_a - f2_b
    pos2 = np.maximum(diff2, 0.0)
    abs2 = np.abs(diff2)
    area_viol2 = float(np.trapezoid(pos2, dx=dx))
    area_total2 = float(np.trapezoid(abs2, dx=dx))
    eps2_star = (area_viol2 / area_total2) if area_total2 > 1e-12 else 0.0
    ssd_strict = bool(np.all(diff2 <= 1e-9) and np.any(diff2 < -1e-6))

    # Tzeng et al. (2013) ASSD condition requires E[R_A] >= E[R_B]
    mean_ok = mean_a >= mean_b
    # Also compute reverse dominance: does benchmark dominate A?
    neg2 = np.maximum(-diff2, 0.0)
    eps2_reverse = (float(np.trapezoid(neg2, dx=dx)) / area_total2) if area_total2 > 1e-12 else 0.0

    return {
        "fsd_strict": fsd_strict,
        "ssd_strict": ssd_strict,
        "eps1_star": round(eps1_star, 4),
        "eps2_star": round(eps2_star, 4),
        "eps2_reverse": round(eps2_reverse, 4),
        "mean_ok": bool(mean_ok),
        "area_viol2": round(area_viol2, 6),
        "area_total2": round(area_total2, 6),
        "f2_a": [round(float(x), 6) for x in f2_a],
        "f2_b": [round(float(x), 6) for x in f2_b],
    }


def classify_preference(sd: dict[str, Any], eps_threshold: float = 0.059) -> str:
    if sd["ssd_strict"]:
        return "YÊU (SSD Chuẩn)"
    if sd["mean_ok"] and sd["eps2_star"] <= eps_threshold:
        return "YÊU (Almost SD)"
    if (not sd["mean_ok"]) and sd["eps2_reverse"] <= eps_threshold:
        return "KHÔNG YÊU (Bị chi phối)"
    return "TRUNG TÍNH (Phân hóa)"


def perf_metrics(returns: np.ndarray, bench_returns: np.ndarray) -> dict[str, float]:
    if len(returns) == 0:
        return {}
    ann_factor = 252.0 / HORIZON_DAYS  # ~11.45 monthly rebalances/yr
    equity = np.cumprod(1.0 + returns)
    peak = np.maximum.accumulate(np.insert(equity, 0, 1.0))[1:]
    dd = (equity - peak) / peak
    mdd = float(dd.min()) if len(dd) else 0.0
    ann_ret = float((equity[-1] ** (ann_factor / len(returns))) - 1.0)
    ann_vol = float(np.std(returns, ddof=1) * np.sqrt(ann_factor)) if len(returns) > 1 else 0.0
    sharpe = ann_ret / ann_vol if ann_vol > 1e-9 else 0.0

    downside = np.minimum(returns, 0.0)
    down_std = float(np.sqrt(np.mean(downside ** 2)) * np.sqrt(ann_factor))
    sortino = ann_ret / down_std if down_std > 1e-9 else 0.0

    gains = float(np.sum(np.maximum(returns, 0.0)))
    losses = float(np.sum(np.abs(np.minimum(returns, 0.0))))
    omega = gains / losses if losses > 1e-9 else 0.0
    win_rate = float(np.mean(returns > bench_returns))
    return {
        "cum_return_pct": round(float(equity[-1] - 1.0) * 100.0, 2),
        "ann_return_pct": round(ann_ret * 100.0, 2),
        "ann_vol_pct": round(ann_vol * 100.0, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "omega": round(omega, 2),
        "max_drawdown_pct": round(mdd * 100.0, 2),
        "beat_bench_rate_pct": round(win_rate * 100.0, 1),
    }


def run_full_pipeline(force_refresh: bool = False) -> dict[str, Any]:
    bundle = fetch_yahoo_vn_data(force_refresh=force_refresh)
    panel, audit_events = sanitize_and_build_panel(bundle)

    feature_cols = [
        "ret_5d",
        "ret_22d",
        "ret_63d",
        "rel_strength_22d",
        "ewma_vol",
        "skew_22d",
        "max_ret_22d",
        "cmf_20",
        "ab_turn",
        "flow_score",
    ]

    all_dates = sorted(panel["date"].unique())
    # Warmup 65 days for rolling features, need horizon for targets, walk-forward monthly steps
    valid_dates = all_dates[70:-HORIZON_DAYS]
    rebalance_dates = valid_dates[260::HORIZON_DAYS]  # ~1 yr warmup train + ~2 yrs OOS monthly steps

    # Walk-forward Backtest storage
    bt_dates: list[str] = []
    ret_assd_ml: list[float] = []
    ret_mean_ml: list[float] = []
    ret_min_vol: list[float] = []
    ret_bench: list[float] = []
    prev_assd_set: set[str] = set()
    prev_mean_set: set[str] = set()
    prev_minv_set: set[str] = set()

    # Track alignment between ML-ASSD score at t and future flow/return at t+1
    flow_alignment_records: list[dict[str, float]] = []

    for reb_idx, reb_date in enumerate(rebalance_dates):
        train_df = panel[
            (panel["date"] < reb_date)
            & panel[feature_cols + ["fwd_ret_22d"]].notna().all(axis=1)
        ]
        # Ensure strict causal cutoff: target at t uses t+22, so train rows must have date <= all_dates[idx - HORIZON_DAYS]
        reb_pos = all_dates.index(reb_date)
        cutoff_date = all_dates[max(0, reb_pos - HORIZON_DAYS)]
        train_df = train_df[train_df["date"] <= cutoff_date]
        test_df = panel[(panel["date"] == reb_date) & panel[feature_cols + ["fwd_ret_22d"]].notna().all(axis=1)]

        if len(train_df) < 300 or len(test_df) < 15:
            continue

        X_tr = train_df[feature_cols].to_numpy(dtype=float)
        y_tr = train_df["fwd_ret_22d"].to_numpy(dtype=float)
        X_te = test_df[feature_cols].to_numpy(dtype=float)

        qrf = QuantileRandomForest(n_trees=45, max_depth=4, min_samples_leaf=30, seed=42 + reb_idx)
        qrf.fit(X_tr, y_tr)
        cdfs, pred_means, _ = qrf.predict_cdfs(X_te, CDF_GRID)

        # Benchmark predicted distribution = cross-sectional mixture of predicted stock CDFs
        bench_cdf = np.mean(cdfs, axis=0)
        bench_mean = float(np.mean(pred_means))

        # Mean-ML baseline (Ridge Regression in pure NumPy)
        X_tr_std = (X_tr - X_tr.mean(axis=0)) / (X_tr.std(axis=0) + 1e-8)
        X_te_std = (X_te - X_tr.mean(axis=0)) / (X_tr.std(axis=0) + 1e-8)
        X_tr_b = np.column_stack([np.ones(len(X_tr_std)), X_tr_std])
        X_te_b = np.column_stack([np.ones(len(X_te_std)), X_te_std])
        reg = 50.0 * np.eye(X_tr_b.shape[1])
        reg[0, 0] = 0.0
        beta = np.linalg.solve(X_tr_b.T @ X_tr_b + reg, X_tr_b.T @ y_tr)
        ridge_preds = X_te_b @ beta

        scored = []
        for idx_row, (_, row) in enumerate(test_df.iterrows()):
            sd = compute_sd_metrics(cdfs[idx_row], bench_cdf, float(pred_means[idx_row]), bench_mean, CDF_GRID)
            # Composite attractiveness score: favor higher mean and smaller violation ratio eps2*
            asd_rank_score = (1.0 - sd["eps2_star"]) * (1.0 + float(pred_means[idx_row]))
            scored.append(
                {
                    "ticker": row["ticker"],
                    "eps2_star": sd["eps2_star"],
                    "mean_ok": sd["mean_ok"],
                    "pred_mean": float(pred_means[idx_row]),
                    "ridge_pred": float(ridge_preds[idx_row]),
                    "ewma_vol": float(row["ewma_vol"]),
                    "actual_fwd_ret": float(row["fwd_ret_22d"]),
                    "actual_fwd_cmf": float(row["fwd_cmf_20"]) if pd.notna(row["fwd_cmf_20"]) else 0.0,
                    "actual_fwd_abturn": float(row["fwd_ab_turn"]) if pd.notna(row["fwd_ab_turn"]) else 1.0,
                    "asd_rank_score": asd_rank_score,
                }
            )
            flow_alignment_records.append(
                {
                    "is_assd_preferred": 1.0 if (sd["mean_ok"] and sd["eps2_star"] <= 0.059) else 0.0,
                    "actual_fwd_ret": float(row["fwd_ret_22d"]),
                    "actual_fwd_cmf": float(row["fwd_cmf_20"]) if pd.notna(row["fwd_cmf_20"]) else 0.0,
                    "actual_fwd_abturn": float(row["fwd_ab_turn"]) if pd.notna(row["fwd_ab_turn"]) else 1.0,
                }
            )

        # Select Top 5 portfolios
        top_k = 5
        assd_candidates = sorted(scored, key=lambda x: (-int(x["mean_ok"] and x["eps2_star"] <= 0.059), x["eps2_star"], -x["pred_mean"]))[:top_k]
        mean_candidates = sorted(scored, key=lambda x: -x["ridge_pred"])[:top_k]
        minv_candidates = sorted(scored, key=lambda x: x["ewma_vol"])[:top_k]

        def net_ret(cands: list[dict[str, Any]], prev_set: set[str]) -> tuple[float, set[str]]:
            cur_set = {c["ticker"] for c in cands}
            turnover = len(cur_set - prev_set) / top_k if prev_set else 1.0
            gross = float(np.mean([c["actual_fwd_ret"] for c in cands]))
            return gross - turnover * TRANSACTION_COST_ROUNDTRIP, cur_set

        r_a, prev_assd_set = net_ret(assd_candidates, prev_assd_set)
        r_m, prev_mean_set = net_ret(mean_candidates, prev_mean_set)
        r_v, prev_minv_set = net_ret(minv_candidates, prev_minv_set)
        r_b = float(np.mean([c["actual_fwd_ret"] for c in scored]))

        bt_dates.append(reb_date)
        ret_assd_ml.append(r_a)
        ret_mean_ml.append(r_m)
        ret_min_vol.append(r_v)
        ret_bench.append(r_b)

    # Latest live prediction (at most recent trading day in dataset!)
    latest_date = all_dates[-1]
    latest_cutoff = all_dates[-1 - HORIZON_DAYS]
    train_latest = panel[
        (panel["date"] <= latest_cutoff)
        & panel[feature_cols + ["fwd_ret_22d"]].notna().all(axis=1)
    ]
    latest_df = panel[(panel["date"] == latest_date) & panel[feature_cols].notna().all(axis=1)].copy()

    qrf_live = QuantileRandomForest(n_trees=70, max_depth=4, min_samples_leaf=30, seed=2026)
    qrf_live.fit(train_latest[feature_cols].to_numpy(dtype=float), train_latest["fwd_ret_22d"].to_numpy(dtype=float))
    live_cdfs, live_means, live_quants = qrf_live.predict_cdfs(latest_df[feature_cols].to_numpy(dtype=float), CDF_GRID)

    bench_live_cdf = np.mean(live_cdfs, axis=0)
    bench_live_mean = float(np.mean(live_means))
    bench_live_quants = np.mean(live_quants, axis=0)

    stocks_live: list[dict[str, Any]] = []
    for idx_row, (_, row) in enumerate(latest_df.iterrows()):
        cdf_i = live_cdfs[idx_row]
        mean_i = float(live_means[idx_row])
        sd = compute_sd_metrics(cdf_i, bench_live_cdf, mean_i, bench_live_mean, CDF_GRID)
        label_default = classify_preference(sd, eps_threshold=0.059)
        q_vals = live_quants[idx_row]
        downside_prob = float(np.interp(0.0, CDF_GRID, cdf_i))
        cvar5 = float(np.mean([q_vals[0], q_vals[1]]))  # average of 5% & 10% left tail

        stocks_live.append(
            {
                "ticker": row["ticker"],
                "short_ticker": row["ticker"].replace(".VN", ""),
                "name": row["name"],
                "sector": row["sector"],
                "last_close_vnd": round(float(row["close_raw"]), 0),
                "ret_5d_pct": round(float(row["ret_5d"]) * 100.0, 2),
                "ret_22d_pct": round(float(row["ret_22d"]) * 100.0, 2),
                "ewma_vol_pct": round(float(row["ewma_vol"]) * 100.0, 2),
                "max_ret_22d_pct": round(float(row["max_ret_22d"]) * 100.0, 2),
                "cmf_20": round(float(row["cmf_20"]), 3),
                "ab_turn": round(float(row["ab_turn"]), 2),
                "flow_score": round(float(row["flow_score"]), 3),
                "pred_mean_pct": round(mean_i * 100.0, 2),
                "bench_pred_mean_pct": round(bench_live_mean * 100.0, 2),
                "loss_prob_pct": round(downside_prob * 100.0, 1),
                "cvar_tail_pct": round(cvar5 * 100.0, 2),
                "quantiles_pct": [round(float(v) * 100.0, 2) for v in q_vals],
                "cdf": [round(float(v), 4) for v in cdf_i],
                "sd": sd,
                "preference_default": label_default,
            }
        )

    stocks_live.sort(key=lambda x: (x["sd"]["eps2_star"], -x["pred_mean_pct"]))

    # Sector Capital Flow Summary
    sector_map: dict[str, list[dict[str, Any]]] = {}
    for s in stocks_live:
        sector_map.setdefault(s["sector"], []).append(s)

    sector_summary: list[dict[str, Any]] = []
    for sec, items in sector_map.items():
        n_pref = sum(1 for x in items if "YÊU" in x["preference_default"] and "KHÔNG" not in x["preference_default"])
        sector_summary.append(
            {
                "sector": sec,
                "count": len(items),
                "preferred_count": n_pref,
                "preferred_ratio_pct": round(100.0 * n_pref / len(items), 1),
                "avg_eps2_star_pct": round(float(np.mean([x["sd"]["eps2_star"] for x in items])) * 100.0, 2),
                "avg_pred_ret_pct": round(float(np.mean([x["pred_mean_pct"] for x in items])), 2),
                "avg_ab_turn": round(float(np.mean([x["ab_turn"] for x in items])), 2),
                "avg_cmf_20": round(float(np.mean([x["cmf_20"] for x in items])), 3),
                "avg_flow_score": round(float(np.mean([x["flow_score"] for x in items])), 3),
                "tickers": [x["short_ticker"] for x in items],
            }
        )
    sector_summary.sort(key=lambda x: (-x["preferred_ratio_pct"], x["avg_eps2_star_pct"]))

    # Flow Alignment Empirical Result across OOS periods
    fa_df = pd.DataFrame(flow_alignment_records)
    pref_grp = fa_df[fa_df["is_assd_preferred"] == 1.0]
    non_grp = fa_df[fa_df["is_assd_preferred"] == 0.0]

    def equity_curve(rets: list[float]) -> list[float]:
        eq = 100.0 * np.cumprod(1.0 + np.array(rets))
        return [100.0] + [round(float(x), 2) for x in eq]

    result = {
        "generated_at": dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "data_source": bundle["source"],
        "date_range": {"start": all_dates[0], "end": latest_date, "trading_days": len(all_dates)},
        "grid_pct": [round(float(x) * 100.0, 2) for x in CDF_GRID],
        "quantile_probs_pct": [round(float(p) * 100.0, 0) for p in QUANTILE_PROBS],
        "benchmark": {
            "name": "VN-Leaders Basket (21 cổ phiếu HOSE)",
            "pred_mean_pct": round(bench_live_mean * 100.0, 2),
            "cdf": [round(float(v), 4) for v in bench_live_cdf],
            "quantiles_pct": [round(float(v) * 100.0, 2) for v in bench_live_quants],
        },
        "stocks": stocks_live,
        "sectors": sector_summary,
        "oos_flow_validation": {
            "n_stock_months": int(len(fa_df)),
            "preferred_avg_fwd_ret_pct": round(float(pref_grp["actual_fwd_ret"].mean()) * 100.0, 2) if len(pref_grp) else 0.0,
            "non_preferred_avg_fwd_ret_pct": round(float(non_grp["actual_fwd_ret"].mean()) * 100.0, 2) if len(non_grp) else 0.0,
            "preferred_avg_fwd_cmf": round(float(pref_grp["actual_fwd_cmf"].mean()), 3) if len(pref_grp) else 0.0,
            "non_preferred_avg_fwd_cmf": round(float(non_grp["actual_fwd_cmf"].mean()), 3) if len(non_grp) else 0.0,
            "preferred_avg_fwd_abturn": round(float(pref_grp["actual_fwd_abturn"].mean()), 2) if len(pref_grp) else 1.0,
            "non_preferred_avg_fwd_abturn": round(float(non_grp["actual_fwd_abturn"].mean()), 2) if len(non_grp) else 1.0,
        },
        "backtest": {
            "dates": ["Khởi đầu"] + bt_dates,
            "n_rebalances": len(bt_dates),
            "cost_note": "Đã khấu trừ 0,35% mỗi vòng xoay danh mục (0,25% phí 2 chiều + 0,10% thuế bán)",
            "curves": {
                "ASSD + QRF (Top-5 Yêu)": equity_curve(ret_assd_ml),
                "Mean-ML (Ridge Top-5)": equity_curve(ret_mean_ml),
                "Min-Volatility (Top-5)": equity_curve(ret_min_vol),
                "VN-Basket Benchmark": equity_curve(ret_bench),
            },
            "metrics": {
                "ASSD + QRF (Top-5 Yêu)": perf_metrics(np.array(ret_assd_ml), np.array(ret_bench)),
                "Mean-ML (Ridge Top-5)": perf_metrics(np.array(ret_mean_ml), np.array(ret_bench)),
                "Min-Volatility (Top-5)": perf_metrics(np.array(ret_min_vol), np.array(ret_bench)),
                "VN-Basket Benchmark": perf_metrics(np.array(ret_bench), np.array(ret_bench)),
            },
        },
        "data_audit": {
            "corporate_action_repairs_count": len(audit_events),
            "sample_repairs": audit_events[:12],
        },
    }
    MODEL_OUTPUT_PATH.write_text(json.dumps(result, ensure_ascii=False), encoding="utf-8")
    return result


if __name__ == "__main__":
    out = run_full_pipeline(force_refresh=False)
    print(f"Pipeline OK! Range: {out['date_range']} | Latest stocks: {len(out['stocks'])}")
    print("Backtest metrics:")
    for k, v in out["backtest"]["metrics"].items():
        print(f"  {k}: {v}")
    print("OOS Flow validation:", out["oos_flow_validation"])
    print("Top 5 stocks by eps2*:")
    for s in out["stocks"][:5]:
        print(f"  {s['short_ticker']:5s} ({s['sector']}) | eps2*={s['sd']['eps2_star']*100:.2f}% | pred_mean={s['pred_mean_pct']:.2f}% | CMF={s['cmf_20']} | {s['preference_default']}")
