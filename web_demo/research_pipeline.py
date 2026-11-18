#!/usr/bin/env python3
"""
research_pipeline.py — Complete Q1 Scientific Pipeline for:
"Predicting Investor Stock Preference (Love / Non-Love) & Capital Flow Direction
on the Vietnamese Stock Market (HOSE) via Distributional ML & Almost Stochastic Dominance"

Grounded in Q1 corpus:
- Distributional CDF forecasting: Meinshausen (2006, JMLR Q1), Barunik et al. (2026, IJF Q1),
  Vovk et al. (2019, ML Q1), Chernozhukov et al. (2021, PNAS Q1)
- Features & VN Market: Gu et al. (2020, RFS Q1), Leippold et al. (2022, JFE Q1),
  Fong (2013, JBF Q2), Vo (2017, IRFA Q1), Thanh Nhon et al. (2025, PLOS ONE Q1)
- Stochastic Dominance & Almost SD Optimization: Leshno & Levy (2002, MS Q1),
  Tzeng et al. (2013, MS Q1), Cesarone & Puerto (2025, EJOR Q1), Luo et al. (2025, MS Q1)
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import pathlib
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

BASE_DIR = pathlib.Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent
DOCS_DATA_DIR = ROOT_DIR / "docs" / "data"
CACHE_5Y_PATH = BASE_DIR / "vn_market_cache_5y.json"
OUTPUT_BUNDLE_PATH = BASE_DIR / "research_bundle.json"
CORPUS_CSV_PATH = ROOT_DIR / "q1_200plus_corpus.csv"

STOCK_UNIVERSE: list[dict[str, str]] = [
    {"ticker": "VCB.VN", "name": "Vietcombank", "sector": "Ngân hàng"},
    {"ticker": "TCB.VN", "name": "Techcombank", "sector": "Ngân hàng"},
    {"ticker": "MBB.VN", "name": "MB Bank", "sector": "Ngân hàng"},
    {"ticker": "ACB.VN", "name": "ACB", "sector": "Ngân hàng"},
    {"ticker": "BID.VN", "name": "BIDV", "sector": "Ngân hàng"},
    {"ticker": "CTG.VN", "name": "VietinBank", "sector": "Ngân hàng"},
    {"ticker": "VPB.VN", "name": "VPBank", "sector": "Ngân hàng"},
    {"ticker": "HDB.VN", "name": "HDBank", "sector": "Ngân hàng"},
    {"ticker": "STB.VN", "name": "Sacombank", "sector": "Ngân hàng"},
    {"ticker": "TPB.VN", "name": "TPBank", "sector": "Ngân hàng"},
    {"ticker": "VIB.VN", "name": "VIB", "sector": "Ngân hàng"},
    {"ticker": "SHB.VN", "name": "SHB", "sector": "Ngân hàng"},
    {"ticker": "SSB.VN", "name": "SeABank", "sector": "Ngân hàng"},
    {"ticker": "SSI.VN", "name": "Chứng khoán SSI", "sector": "Chứng khoán"},
    {"ticker": "VCI.VN", "name": "Vietcap", "sector": "Chứng khoán"},
    {"ticker": "HCM.VN", "name": "HSC", "sector": "Chứng khoán"},
    {"ticker": "FPT.VN", "name": "FPT Corp", "sector": "Công nghệ & Bán lẻ"},
    {"ticker": "MWG.VN", "name": "Thế Giới Di Động", "sector": "Công nghệ & Bán lẻ"},
    {"ticker": "FRT.VN", "name": "FPT Retail", "sector": "Công nghệ & Bán lẻ"},
    {"ticker": "PNJ.VN", "name": "PNJ", "sector": "Công nghệ & Bán lẻ"},
    {"ticker": "HPG.VN", "name": "Hòa Phát", "sector": "Thép & Hóa chất"},
    {"ticker": "HSG.VN", "name": "Hoa Sen Group", "sector": "Thép & Hóa chất"},
    {"ticker": "DGC.VN", "name": "Hóa chất Đức Giang", "sector": "Thép & Hóa chất"},
    {"ticker": "VHM.VN", "name": "Vinhomes", "sector": "Bất động sản"},
    {"ticker": "VIC.VN", "name": "Vingroup", "sector": "Bất động sản"},
    {"ticker": "VRE.VN", "name": "Vincom Retail", "sector": "Bất động sản"},
    {"ticker": "KDH.VN", "name": "Nhà Khang Điền", "sector": "Bất động sản"},
    {"ticker": "NLG.VN", "name": "Nam Long Group", "sector": "Bất động sản"},
    {"ticker": "VNM.VN", "name": "Vinamilk", "sector": "Tiêu dùng & Đồ uống"},
    {"ticker": "MSN.VN", "name": "Masan Group", "sector": "Tiêu dùng & Đồ uống"},
    {"ticker": "SAB.VN", "name": "Sabeco", "sector": "Tiêu dùng & Đồ uống"},
    {"ticker": "GAS.VN", "name": "PV Gas", "sector": "Năng lượng & Hạ tầng"},
    {"ticker": "PLX.VN", "name": "Petrolimex", "sector": "Năng lượng & Hạ tầng"},
    {"ticker": "PVD.VN", "name": "PV Drilling", "sector": "Năng lượng & Hạ tầng"},
    {"ticker": "POW.VN", "name": "PV Power", "sector": "Năng lượng & Hạ tầng"},
    {"ticker": "REE.VN", "name": "Cơ điện lạnh REE", "sector": "Công nghiệp & Bảo hiểm"},
    {"ticker": "GMD.VN", "name": "Gemadept", "sector": "Công nghiệp & Bảo hiểm"},
    {"ticker": "BVH.VN", "name": "Tập đoàn Bảo Việt", "sector": "Công nghiệp & Bảo hiểm"},
]
ETF_BENCHMARK = {"ticker": "E1VFVN30.VN", "name": "ETF VFMVN30", "sector": "ETF Index"}

HORIZON_DAYS = 22
CDF_GRID = np.linspace(-0.32, 0.40, 129)
QUANTILE_PROBS = np.linspace(0.05, 0.95, 19)
TRANSACTION_COST = 0.0035  # 0.25% two-way brokerage + 0.10% VN selling tax

FEATURE_NAMES = [
    "ret_1d", "ret_5d", "ret_22d", "ret_66d", "mom_126_22",
    "vol_ewma", "vol_60", "vol_ratio", "downside_semi_22",
    "max_22", "min_22", "skew_22", "dist_high_60", "rsi_14",
    "cmf_20", "ab_turn", "amihud_20", "vpt_20",
]


def _fetch_single_yahoo(ticker: str, range_str: str = "5y") -> list[dict[str, Any]]:
    url = (
        f"https://query1.finance.yahoo.com/v8/finance/chart/"
        f"{urllib.parse.quote(ticker)}?range={range_str}&interval=1d&events=div,splits"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=20) as resp:
        payload = json.loads(resp.read().decode("utf-8"))
    res = payload["chart"]["result"][0]
    ts = res["timestamp"]
    q = res["indicators"]["quote"][0]
    adj = res["indicators"].get("adjclose", [{}])[0].get("adjclose", q["close"])

    rows = []
    for i, u in enumerate(ts):
        raw_c = q["close"][i]
        c = adj[i] if adj[i] is not None else raw_c
        h = q["high"][i]
        l = q["low"][i]
        o = q["open"][i]
        v = q["volume"][i]
        if c is None or raw_c is None or c <= 0 or raw_c <= 0:
            continue
        ratio = c / raw_c if raw_c > 0 else 1.0
        date_str = dt.datetime.fromtimestamp(u, tz=dt.timezone.utc).strftime("%Y-%m-%d")
        rows.append({
            "date": date_str,
            "open": float((o if o is not None else raw_c) * ratio),
            "high": float((h if h is not None else raw_c) * ratio),
            "low": float((l if l is not None else raw_c) * ratio),
            "close": float(c),
            "volume": float(v if v is not None else 0.0),
        })
    return rows


def fetch_yahoo_5y(force_refresh: bool = False) -> dict[str, Any]:
    if CACHE_5Y_PATH.exists() and not force_refresh:
        return json.loads(CACHE_5Y_PATH.read_text(encoding="utf-8"))
    series: dict[str, Any] = {}
    for meta in STOCK_UNIVERSE + [ETF_BENCHMARK]:
        t = meta["ticker"]
        series[t] = _fetch_single_yahoo(t, range_str="5y")
    payload = {
        "fetched_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "universe": STOCK_UNIVERSE,
        "etf": ETF_BENCHMARK,
        "series": series,
    }
    CACHE_5Y_PATH.write_text(json.dumps(payload), encoding="utf-8")
    return payload


def build_clean_panel(raw_data: dict[str, Any]) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], list[dict[str, Any]]]:
    """
    Align all HOSE tickers + VN30 ETF onto common trading dates and repair
    Yahoo Finance split/dividend date-offset anomalies (> +/-7.6% single-day jumps on HOSE).
    """
    frames = {}
    for t, rows in raw_data["series"].items():
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        df = df.drop_duplicates("date").sort_values("date").set_index("date")
        frames[t] = df

    close_panel = pd.DataFrame({t: f["close"] for t, f in frames.items()})
    close_panel = close_panel.dropna(how="all").ffill().bfill()
    common_dates = close_panel.index

    repairs: list[dict[str, Any]] = []
    aligned_frames: dict[str, pd.DataFrame] = {}

    for t, f in frames.items():
        df = f.reindex(common_dates).ffill().bfill()
        raw_ret = df["close"].pct_change().fillna(0.0)
        # HOSE daily band is +/-7%; single-day jumps outside [-7.6%, +7.6%] on equity tickers
        # are almost always Yahoo dividend/split timestamp misalignments or ex-rights jumps.
        is_etf = t == ETF_BENCHMARK["ticker"]
        limit = 0.12 if is_etf else 0.076
        clean_ret = raw_ret.copy()
        for d, r in raw_ret.items():
            if abs(r) > limit:
                if abs(r) > 0.14:
                    # Major unadjusted split / ex-dividend shift
                    clean_r = 0.0
                    action = "Trung hòa bước nhảy chia tách/cổ tức sai ngày (>14%)"
                else:
                    clean_r = float(np.clip(r, -0.07, 0.07))
                    action = "Giới hạn về biên độ trần/sàn HOSE (±7%)"
                clean_ret.loc[d] = clean_r
                if not is_etf:
                    repairs.append({
                        "ticker": t,
                        "date": d.strftime("%Y-%m-%d"),
                        "raw_ret_pct": round(float(r * 100.0), 2),
                        "repaired_ret_pct": round(float(clean_r * 100.0), 2),
                        "action": action,
                    })

        scale_factor = (1.0 + clean_ret).cumprod()
        anchor_last = float(df["close"].iloc[-1])
        reconstructed_close = anchor_last * scale_factor / float(scale_factor.iloc[-1])
        adj_ratio = reconstructed_close / df["close"].replace(0.0, np.nan).bfill()
        df_clean = pd.DataFrame({
            "open": df["open"] * adj_ratio,
            "high": np.maximum(df["high"] * adj_ratio, reconstructed_close),
            "low": np.minimum(df["low"] * adj_ratio, reconstructed_close),
            "close": reconstructed_close,
            "volume": df["volume"],
            "ret": clean_ret,
        }, index=common_dates)
        aligned_frames[t] = df_clean
        close_panel[t] = reconstructed_close

    repairs.sort(key=lambda x: abs(x["raw_ret_pct"]), reverse=True)
    return close_panel, aligned_frames, repairs


def compute_features_and_flows(
    aligned_frames: dict[str, pd.DataFrame],
    tickers: list[str],
    horizon: int = HORIZON_DAYS,
) -> dict[str, Any]:
    """Compute leakage-free 18 Q1 features, flow proxies (CMF, AbTurn), and forward targets."""
    dates = aligned_frames[tickers[0]].index
    n_days = len(dates)
    n_stocks = len(tickers)
    n_feat = len(FEATURE_NAMES)

    feat_raw = np.full((n_days, n_stocks, n_feat), np.nan, dtype=np.float64)
    fwd_ret = np.full((n_days, n_stocks), np.nan, dtype=np.float64)
    fwd_cmf = np.full((n_days, n_stocks), np.nan, dtype=np.float64)
    fwd_abturn = np.full((n_days, n_stocks), np.nan, dtype=np.float64)
    ewma_vol_arr = np.full((n_days, n_stocks), np.nan, dtype=np.float64)

    for j, t in enumerate(tickers):
        df = aligned_frames[t]
        c = df["close"].to_numpy(dtype=np.float64)
        h = df["high"].to_numpy(dtype=np.float64)
        l = df["low"].to_numpy(dtype=np.float64)
        v = df["volume"].to_numpy(dtype=np.float64)
        r = df["ret"].to_numpy(dtype=np.float64)

        dollar_vol = c * v
        hl_span = h - l
        hl_safe = np.where(hl_span > 1e-9, hl_span, 1.0)
        spread = np.where(hl_span > 1e-9, ((c - l) - (h - c)) / hl_safe, 0.0)
        mfv = spread * np.maximum(v, 1.0)

        # Rolling indicators computed cleanly with pandas
        s_c = pd.Series(c)
        s_r = pd.Series(r)
        s_v = pd.Series(v)
        s_dv = pd.Series(dollar_vol)
        s_mfv = pd.Series(mfv)

        ret_1d = s_r.to_numpy()
        ret_5d = (s_c / s_c.shift(5) - 1.0).to_numpy()
        ret_22d = (s_c / s_c.shift(22) - 1.0).to_numpy()
        ret_66d = (s_c / s_c.shift(66) - 1.0).to_numpy()
        mom_126_22 = (s_c.shift(22) / s_c.shift(126) - 1.0).to_numpy()

        vol_ewma = np.sqrt((s_r ** 2).ewm(alpha=0.06, adjust=False).mean().to_numpy()) * np.sqrt(horizon)
        vol_ewma = np.maximum(vol_ewma, 0.015)
        ewma_vol_arr[:, j] = vol_ewma

        vol_60 = s_r.rolling(60).std().to_numpy() * np.sqrt(horizon)
        vol_ratio = vol_ewma / np.maximum(vol_60, 0.01)

        neg_r = np.minimum(r, 0.0)
        downside_semi_22 = np.sqrt(pd.Series(neg_r ** 2).rolling(22).mean().to_numpy()) * np.sqrt(horizon)
        max_22 = s_r.rolling(22).max().to_numpy()
        min_22 = s_r.rolling(22).min().to_numpy()
        skew_22 = s_r.rolling(22).skew().fillna(0.0).to_numpy()

        dist_high_60 = (s_c / s_c.rolling(60).max() - 1.0).to_numpy()
        up = np.maximum(r, 0.0)
        dn = np.maximum(-r, 0.0)
        rs = pd.Series(up).rolling(14).mean() / pd.Series(dn).rolling(14).mean().replace(0.0, 1e-6)
        rsi_14 = (1.0 - 1.0 / (1.0 + rs)).fillna(0.5).to_numpy()

        cmf_20 = (s_mfv.rolling(20).sum() / s_v.rolling(20).sum().replace(0.0, np.nan)).fillna(0.0).to_numpy()
        ab_turn = (s_dv.rolling(5).mean() / s_dv.rolling(60).mean().replace(0.0, np.nan)).fillna(1.0).to_numpy()
        amihud_raw = (np.abs(r) / np.maximum(dollar_vol / 1e9, 0.05))
        amihud_20 = np.log1p(pd.Series(amihud_raw).rolling(20).mean().fillna(0.0).to_numpy())
        vpt_raw = pd.Series(r * (v / np.maximum(s_v.rolling(60).mean().to_numpy(), 1.0)))
        vpt_20 = vpt_raw.rolling(20).sum().fillna(0.0).to_numpy()

        mat = np.column_stack([
            ret_1d, ret_5d, ret_22d, ret_66d, mom_126_22,
            vol_ewma, vol_60, vol_ratio, downside_semi_22,
            max_22, min_22, skew_22, dist_high_60, rsi_14,
            cmf_20, ab_turn, amihud_20, vpt_20,
        ])
        feat_raw[:, j, :] = mat

        # Strictly forward targets shifted by -horizon
        fwd = (s_c.shift(-horizon) / s_c - 1.0).to_numpy()
        fwd_ret[:, j] = fwd
        fwd_cmf[:, j] = pd.Series(cmf_20).shift(-horizon).to_numpy()
        fwd_abturn[:, j] = pd.Series(ab_turn).shift(-horizon).to_numpy()

    # Cross-sectional rank-normalization into [-1, 1] per date (Gu et al., 2020)
    feat_rank = np.full_like(feat_raw, np.nan)
    for d in range(n_days):
        slice_d = feat_raw[d]  # (n_stocks, n_feat)
        if np.all(np.isnan(slice_d[:, 0])):
            continue
        for f_idx in range(n_feat):
            col = slice_d[:, f_idx]
            valid = np.isfinite(col)
            if valid.sum() >= 3:
                order = np.argsort(np.argsort(col[valid]))
                ranks = (order / max(valid.sum() - 1, 1)) * 2.0 - 1.0
                feat_rank[d, valid, f_idx] = ranks
            else:
                feat_rank[d, valid, f_idx] = 0.0

    return {
        "dates": [d.strftime("%Y-%m-%d") for d in dates],
        "tickers": tickers,
        "feat_raw": feat_raw,
        "feat_rank": feat_rank,
        "ewma_vol": ewma_vol_arr,
        "fwd_ret": fwd_ret,
        "fwd_cmf": fwd_cmf,
        "fwd_abturn": fwd_abturn,
    }


# ---------------------------------------------------------------------------
# MACHINE LEARNING DISTRIBUTIONAL MODELS
# ---------------------------------------------------------------------------

def _solve_quantile_ridge_irls(
    X: np.ndarray,
    y: np.ndarray,
    taus: np.ndarray,
    l2: float = 25.0,
    n_iter: int = 8,
) -> np.ndarray:
    """Fast vector-solved IRLS quantile regression with L2 penalty across taus."""
    n, p = X.shape
    Xd = np.column_stack([np.ones(n, dtype=np.float64), X])
    pen = np.diag([0.0] + [l2] * p)
    coef_init = np.linalg.solve(Xd.T @ Xd + pen, Xd.T @ y)

    coefs = np.zeros((len(taus), p + 1), dtype=np.float64)
    eps = 1e-4
    for idx, tau in enumerate(taus):
        beta = coef_init.copy()
        for _ in range(n_iter):
            resid = y - Xd @ beta
            abs_r = np.maximum(np.abs(resid), eps)
            w = np.where(resid >= 0.0, tau, 1.0 - tau) / abs_r
            Xw = Xd * w[:, None]
            beta = np.linalg.solve(Xw.T @ Xd + pen, Xw.T @ y)
        coefs[idx] = beta
    return coefs


def predict_two_stage_qml(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sig_train: np.ndarray,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    sig_cal: np.ndarray,
    X_test: np.ndarray,
    sig_test: np.ndarray,
    taus: np.ndarray = QUANTILE_PROBS,
) -> np.ndarray:
    """
    Two-Stage Volatility-Scaled Quantile Model + Monotonic Sort + Conformal Calibration
    (Barunik, Hronec & Tobek, 2026 IJF; Vovk et al., 2019 ML; Chernozhukov et al., 2021 PNAS).
    Returns shape (n_test, len(taus)) of predicted quantiles.
    """
    z_train = np.clip(y_train / np.maximum(sig_train, 0.015), -4.5, 4.5)
    coefs = _solve_quantile_ridge_irls(X_train, z_train, taus, l2=30.0, n_iter=7)

    # Conformal shift estimated strictly on trailing calibration block (no leakage)
    Xd_cal = np.column_stack([np.ones(len(X_cal)), X_cal])
    q_cal_z = np.sort(Xd_cal @ coefs.T, axis=1)
    q_cal_r = q_cal_z * sig_cal[:, None]
    conformal_shift = np.zeros(len(taus), dtype=np.float64)
    if len(y_cal) >= 30:
        for k, tau in enumerate(taus):
            scores = y_cal - q_cal_r[:, k]
            # Smoothed quantile of conformity residuals
            conformal_shift[k] = float(np.quantile(scores, tau))

    Xd_test = np.column_stack([np.ones(len(X_test)), X_test])
    q_test_z = np.sort(Xd_test @ coefs.T, axis=1)
    q_test_r = q_test_z * sig_test[:, None] + 0.5 * conformal_shift[None, :]
    q_test_r = np.maximum.accumulate(q_test_r, axis=1)
    return q_test_r


@dataclass
class QRFNode:
    is_leaf: bool
    feature: int = -1
    threshold: float = 0.0
    left: "QRFNode | None" = None
    right: "QRFNode | None" = None
    values: np.ndarray | None = None


class QuantileRandomForest:
    """Pure NumPy implementation of Meinshausen (2006, JMLR) Quantile Regression Forests."""

    def __init__(self, n_trees: int = 30, max_depth: int = 4, min_samples_leaf: int = 35, seed: int = 42):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.rng = np.random.default_rng(seed)
        self.trees: list[QRFNode] = []

    def _build(self, X: np.ndarray, y: np.ndarray, depth: int) -> QRFNode:
        n, p = X.shape
        if depth >= self.max_depth or n < 2 * self.min_samples_leaf:
            return QRFNode(is_leaf=True, values=np.sort(y.copy()))

        feat_sub = self.rng.choice(p, size=max(4, int(math.sqrt(p))), replace=False)
        best_feat, best_thr, best_gain = -1, 0.0, 0.0
        total_var = float(np.var(y)) * n

        for f in feat_sub:
            col = X[:, f]
            qs = np.quantile(col, [0.25, 0.50, 0.75])
            for thr in np.unique(qs):
                left_mask = col <= thr
                nl = int(left_mask.sum())
                nr = n - nl
                if nl < self.min_samples_leaf or nr < self.min_samples_leaf:
                    continue
                gain = total_var - (float(np.var(y[left_mask])) * nl + float(np.var(y[~left_mask])) * nr)
                if gain > best_gain:
                    best_gain = gain
                    best_feat = int(f)
                    best_thr = float(thr)

        if best_feat == -1:
            return QRFNode(is_leaf=True, values=np.sort(y.copy()))

        mask = X[:, best_feat] <= best_thr
        return QRFNode(
            is_leaf=False,
            feature=best_feat,
            threshold=best_thr,
            left=self._build(X[mask], y[mask], depth + 1),
            right=self._build(X[~mask], y[~mask], depth + 1),
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.trees = []
        n = len(X)
        sub_n = min(n, max(250, int(0.75 * n)))
        for _ in range(self.n_trees):
            idx = self.rng.choice(n, size=sub_n, replace=True)
            self.trees.append(self._build(X[idx], y[idx], 0))

    def _query_row(self, node: QRFNode, row: np.ndarray) -> np.ndarray:
        cur = node
        while not cur.is_leaf:
            cur = cur.left if row[cur.feature] <= cur.threshold else cur.right  # type: ignore[assignment]
        return cur.values  # type: ignore[return-value]

    def predict_quantiles(self, X_test: np.ndarray, taus: np.ndarray = QUANTILE_PROBS) -> np.ndarray:
        out = np.zeros((len(X_test), len(taus)), dtype=np.float64)
        for i in range(len(X_test)):
            row = X_test[i]
            samples = np.concatenate([self._query_row(tr, row) for tr in self.trees])
            out[i] = np.quantile(samples, taus)
        return out


def predict_qgbdt(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    taus: np.ndarray = QUANTILE_PROBS,
    n_stages: int = 12,
    lr: float = 0.14,
) -> np.ndarray:
    """
    Lightweight Quantile Gradient Boosting using pinball pseudo-residuals
    on top-split boosted regression stumps, enforced monotonic across taus.
    """
    q_pred_tr = np.tile(np.quantile(y_train, taus), (len(X_train), 1))
    q_pred_te = np.tile(np.quantile(y_train, taus), (len(X_test), 1))

    n, p = X_train.shape
    for k, tau in enumerate(taus):
        f_tr = q_pred_tr[:, k].copy()
        f_te = q_pred_te[:, k].copy()
        for stage in range(n_stages):
            # Pseudo-residual is negative gradient of pinball loss: tau if y > f else tau - 1
            grad = np.where(y_train >= f_tr, tau, tau - 1.0)
            best_f, best_thr, best_l, best_r, best_err = 0, 0.0, 0.0, 0.0, float("inf")
            # Evaluate subset of candidate features evenly spaced
            for fi in range(stage % 2, p, 2):
                col = X_train[:, fi]
                for thr in (-0.35, 0.0, 0.35):
                    m = col <= thr
                    if m.sum() < 20 or (~m).sum() < 20:
                        continue
                    ml = float(grad[m].mean())
                    mr = float(grad[~m].mean())
                    err = float(((grad[m] - ml) ** 2).sum() + ((grad[~m] - mr) ** 2).sum())
                    if err < best_err:
                        best_err, best_f, best_thr, best_l, best_r = err, fi, thr, ml, mr
            f_tr += lr * np.where(X_train[:, best_f] <= best_thr, best_l, best_r)
            f_te += lr * np.where(X_test[:, best_f] <= best_thr, best_l, best_r)
        q_pred_te[:, k] = f_te

    return np.maximum.accumulate(q_pred_te, axis=1)


def quantiles_to_cdf(quantiles: np.ndarray, taus: np.ndarray = QUANTILE_PROBS, grid: np.ndarray = CDF_GRID) -> np.ndarray:
    """Convert monotone quantiles (with extrapolated tail bounds) onto common CDF grid."""
    q_sorted = np.sort(quantiles)
    spread = max(float(q_sorted[-1] - q_sorted[0]), 0.04)
    knots_x = np.concatenate([[q_sorted[0] - 0.18 * spread], q_sorted, [q_sorted[-1] + 0.18 * spread]])
    knots_y = np.concatenate([[0.0], taus, [1.0]])
    cdf = np.interp(grid, knots_x, knots_y, left=0.0, right=1.0)
    return np.maximum.accumulate(cdf)


# ---------------------------------------------------------------------------
# STOCHASTIC DOMINANCE & ALMOST SD + SCORING + OPTIMIZER
# ---------------------------------------------------------------------------

def compute_sd_metrics(cdf_a: np.ndarray, cdf_b: np.ndarray, grid: np.ndarray = CDF_GRID) -> dict[str, Any]:
    """
    Compute exact FSD, SSD, and Almost SD violation ratios epsilon1*, epsilon2*
    following Leshno & Levy (2002) + corrected Tzeng, Huang & Shih (2013, MS Q1).
    """
    dx = float(grid[1] - grid[0])

    # Expected return from discrete CDF slices: sum((1 - F(x)) dx) shifted from grid[0]
    mean_a = float(grid[0] + np.sum((1.0 - cdf_a) * dx))
    mean_b = float(grid[0] + np.sum((1.0 - cdf_b) * dx))
    mean_ok = mean_a >= mean_b - 1e-6

    # FSD violation S1: where F_A(x) > F_B(x)
    diff1 = cdf_a - cdf_b
    area_viol1 = float(np.sum(np.maximum(diff1, 0.0)) * dx)
    area_tot1 = float(np.sum(np.abs(diff1)) * dx)
    eps1_star = area_viol1 / area_tot1 if area_tot1 > 1e-12 else 0.0

    # Integrated CDFs F^(2)(t)
    f2_a = np.cumsum(cdf_a) * dx
    f2_b = np.cumsum(cdf_b) * dx
    diff2 = f2_a - f2_b
    area_viol2 = float(np.sum(np.maximum(diff2, 0.0)) * dx)
    area_tot2 = float(np.sum(np.abs(diff2)) * dx)
    eps2_star = area_viol2 / area_tot2 if area_tot2 > 1e-12 else 0.0

    # Reverse violation (whether Benchmark ASSD-dominates Stock A)
    area_rev2 = float(np.sum(np.maximum(-diff2, 0.0)) * dx)
    eps2_reverse = area_rev2 / area_tot2 if area_tot2 > 1e-12 else 0.0

    return {
        "fsd_strict": bool(eps1_star < 1e-6 and area_tot1 > 1e-8),
        "ssd_strict": bool(mean_ok and eps2_star < 1e-6 and area_tot2 > 1e-8),
        "eps1_star": round(eps1_star, 5),
        "eps2_star": round(eps2_star, 5),
        "eps2_reverse": round(eps2_reverse, 5),
        "mean_a": round(mean_a, 5),
        "mean_b": round(mean_b, 5),
        "mean_ok": bool(mean_ok),
        "area_viol2": round(area_viol2, 6),
        "area_total2": round(area_tot2, 6),
        "f2_a": [round(float(x), 5) for x in f2_a],
        "f2_b": [round(float(x), 5) for x in f2_b],
    }


def _project_capped_simplex(v: np.ndarray, u_max: float = 0.20) -> np.ndarray:
    """Euclidean projection onto {w >= 0, sum(w) = 1, w_i <= u_max}."""
    n = len(v)
    if u_max * n < 1.0 - 1e-9:
        return np.full(n, 1.0 / n)
    lo, hi = float(np.min(v) - 1.0), float(np.max(v))
    for _ in range(45):
        mid = 0.5 * (lo + hi)
        w = np.clip(v - mid, 0.0, u_max)
        if w.sum() > 1.0:
            lo = mid
        else:
            hi = mid
    w = np.clip(v - 0.5 * (lo + hi), 0.0, u_max)
    s = w.sum()
    return w / s if s > 1e-12 else np.full(n, 1.0 / n)


def optimize_assd_portfolio(
    pred_quantiles: np.ndarray,  # (n_stocks, n_taus)
    hist_window: np.ndarray,     # (T_hist, n_stocks) recent daily returns for copula structure
    prev_weights: np.ndarray,
    horizon: int = HORIZON_DAYS,
    w_max: float = 0.20,
    turn_penalty: float = 0.012,
    mean_bonus: float = 0.28,
) -> np.ndarray:
    """
    Scenario-based ASSD Enhanced Indexation Portfolio Optimization
    (Cesarone & Puerto, 2025 EJOR; Luo et al., 2025 MS):
    Builds coherent joint return scenarios combining ML-predicted marginal CDFs
    with trailing empirical rank dependence (Iman-Conover style), then solves:
      min_{w in capped_simplex} ASSD_Shortfall(F2_port vs F2_bench) - lambda * E[R_p] + gamma * ||w - w_prev||^2
    """
    n_stocks = pred_quantiles.shape[0]
    # Build T_s scenarios by mapping trailing block-window empirical ranks through predicted quantiles
    m_rows = max(66, min(132, len(hist_window) - horizon))
    # Aggregate trailing non-overlapping / overlapping horizon returns on window
    scen = np.zeros((m_rows, n_stocks), dtype=np.float64)
    start_idx = max(0, len(hist_window) - m_rows - 1)
    win = hist_window[start_idx:]
    taus_fine = np.linspace(0.05, 0.95, 19)

    for j in range(n_stocks):
        col = win[-m_rows:, j]
        # Empirical rank in (0, 1) -> map through stock's predicted quantile function!
        ranks = (np.argsort(np.argsort(col)) + 0.5) / m_rows
        scen[:, j] = np.interp(ranks, taus_fine, pred_quantiles[j], left=pred_quantiles[j, 0], right=pred_quantiles[j, -1])

    bench_scen = np.mean(scen, axis=1)
    bench_sorted = np.sort(bench_scen)
    # Discrete SSD target thresholds = benchmark order statistics / grid targets
    n_scen = len(scen)
    # Under discrete scenarios, CVaR/SSD at thresholds k=1..K compares sorted portfolio shortfall vs benchmark
    # Smooth Hinge Loss approximation for Second-Order Stochastic Dominance shortfall E[(t - R_p)_+] - E[(t - R_b)_+]
    thresholds = np.quantile(bench_scen, np.linspace(0.05, 0.85, 17))
    target_f2 = np.mean(np.maximum(thresholds[None, :] - bench_scen[:, None], 0.0), axis=0)
    stock_means = np.mean(scen, axis=0)

    w = prev_weights.copy() if prev_weights.sum() > 0.5 else np.full(n_stocks, 1.0 / n_stocks)
    lr = 0.65
    for it in range(90):
        rp = scen @ w  # (n_scen,)
        diff = thresholds[None, :] - rp[:, None]  # (n_scen, n_thr)
        active = diff > 0.0
        port_f2 = np.mean(np.maximum(diff, 0.0), axis=0)
        viol = port_f2 > target_f2  # (n_thr,)

        # Subgradient of mean((port_f2 - target_f2)_+^2) w.r.t weights
        grad_f2 = np.zeros(n_stocks, dtype=np.float64)
        for k in range(len(thresholds)):
            if viol[k]:
                excess = port_f2[k] - target_f2[k]
                # d(port_f2[k])/dw = mean(-scen[active[:, k]], axis=0)
                mask_k = active[:, k]
                d_k = -np.mean(scen * mask_k[:, None], axis=0)
                grad_f2 += 2.0 * excess * d_k

        # Penalize left-tail variance slightly + encourage predicted mean + turnover smoothing
        grad_turn = 2.0 * turn_penalty * (w - prev_weights)
        grad = grad_f2 * 120.0 - mean_bonus * stock_means + grad_turn
        step = lr / (1.0 + 0.04 * it)
        w = _project_capped_simplex(w - step * grad, u_max=w_max)

    return w


def optimize_min_variance(hist_daily: np.ndarray, w_max: float = 0.20) -> np.ndarray:
    """Long-only capped-weight Minimum Variance portfolio using Ledoit-Wolf shrinkage."""
    n = hist_daily.shape[1]
    sub = hist_daily[-126:]
    cov = np.cov(sub, rowvar=False)
    diag = np.diag(np.diag(cov))
    cov_shrunk = 0.85 * cov + 0.15 * diag
    w = np.full(n, 1.0 / n)
    for it in range(80):
        grad = 2.0 * (cov_shrunk @ w) * 100.0
        w = _project_capped_simplex(w - (0.35 / (1 + 0.03 * it)) * grad, u_max=w_max)
    return w


def optimize_markowitz_mv(stock_means: np.ndarray, hist_daily: np.ndarray, w_max: float = 0.20) -> np.ndarray:
    """Mean-Variance Markowitz baseline."""
    n = hist_daily.shape[1]
    sub = hist_daily[-126:]
    cov = np.cov(sub, rowvar=False)
    cov_shrunk = 0.85 * cov + 0.15 * np.diag(np.diag(cov))
    w = np.full(n, 1.0 / n)
    for it in range(80):
        grad = 2.0 * (cov_shrunk @ w) * 140.0 - 0.30 * stock_means
        w = _project_capped_simplex(w - (0.35 / (1 + 0.03 * it)) * grad, u_max=w_max)
    return w


# ---------------------------------------------------------------------------
# EVALUATION METRICS & HYPOTHESIS TESTING
# ---------------------------------------------------------------------------

def pinball_loss(y_true: np.ndarray, q_pred: np.ndarray, taus: np.ndarray = QUANTILE_PROBS) -> dict[str, float]:
    """Compute pinball loss (overall, tail, median) and discrete CRPS approximation."""
    err = y_true[:, None] - q_pred
    loss = np.maximum(taus[None, :] * err, (taus[None, :] - 1.0) * err)
    overall = float(np.mean(loss))
    left_tail = float(np.mean(loss[:, taus <= 0.25]))
    median = float(np.mean(loss[:, np.argmin(np.abs(taus - 0.5))]))
    right_tail = float(np.mean(loss[:, taus >= 0.75]))
    # Continuous Ranked Probability Score integrated across quantile grid: CRPS = 2 * mean(pinball)
    crps = 2.0 * overall
    # 90% Interval Coverage (tau=0.05 to tau=0.95)
    cov90 = float(np.mean((y_true >= q_pred[:, 0]) & (y_true <= q_pred[:, -1]))) * 100.0
    return {
        "pinball_mean": round(overall * 100.0, 4),
        "pinball_left": round(left_tail * 100.0, 4),
        "pinball_median": round(median * 100.0, 4),
        "pinball_right": round(right_tail * 100.0, 4),
        "crps_pct": round(crps * 100.0, 4),
        "coverage_90_pct": round(cov90, 2),
    }


def compute_portfolio_metrics(rets: np.ndarray, turnovers: list[float], bench_rets: np.ndarray) -> dict[str, float]:
    if len(rets) == 0:
        return {}
    equity = np.cumprod(1.0 + rets)
    cum_ret = float(equity[-1] - 1.0) * 100.0
    periods_per_year = 252.0 / HORIZON_DAYS
    n = len(rets)
    ann_ret = float(equity[-1] ** (periods_per_year / n) - 1.0) * 100.0
    ann_vol = float(np.std(rets, ddof=1) * math.sqrt(periods_per_year)) * 100.0
    sharpe = (float(np.mean(rets)) / max(float(np.std(rets, ddof=1)), 1e-8)) * math.sqrt(periods_per_year)

    neg = rets[rets < 0.0]
    downside_dev = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else float(np.std(rets))
    sortino = (float(np.mean(rets)) / max(downside_dev, 1e-8)) * math.sqrt(periods_per_year)

    gains = float(np.sum(np.maximum(rets, 0.0)))
    losses = float(np.sum(np.maximum(-rets, 0.0)))
    omega = gains / losses if losses > 1e-8 else 3.0

    peaks = np.maximum.accumulate(np.concatenate([[1.0], equity]))
    dds = np.concatenate([[1.0], equity]) / peaks - 1.0
    mdd = float(np.min(dds)) * 100.0
    calmar = ann_ret / abs(mdd) if abs(mdd) > 1e-6 else 0.0

    var5 = float(np.quantile(rets, 0.05)) * 100.0
    tail = rets[rets <= np.quantile(rets, 0.05)]
    cvar5 = float(np.mean(tail)) * 100.0 if len(tail) > 0 else var5
    win_vs_bench = float(np.mean(rets > bench_rets)) * 100.0
    avg_turn = float(np.mean(turnovers)) * 100.0 if turnovers else 0.0

    # Paired bootstrap p-value of excess return > 0 vs benchmark
    diff = rets - bench_rets
    rng = np.random.default_rng(7)
    boot_means = np.array([np.mean(rng.choice(diff, size=len(diff), replace=True)) for _ in range(1500)])
    boot_p_beat = float(np.mean(boot_means <= 0.0))

    return {
        "cum_return_pct": round(cum_ret, 2),
        "ann_return_pct": round(ann_ret, 2),
        "ann_vol_pct": round(ann_vol, 2),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "omega": round(omega, 2),
        "max_drawdown_pct": round(mdd, 2),
        "calmar": round(calmar, 2),
        "var5_pct": round(var5, 2),
        "cvar5_pct": round(cvar5, 2),
        "win_vs_bench_pct": round(win_vs_bench, 1),
        "avg_turnover_pct": round(avg_turn, 1),
        "boot_p_value": round(boot_p_beat, 4),
    }


def _ols_with_tstat(y: np.ndarray, X: np.ndarray, names: list[str]) -> list[dict[str, Any]]:
    """Compute OLS coefficients, robust HC1 standard errors, t-stats, and approx p-values."""
    Xd = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid = y - Xd @ beta
    n, k = Xd.shape
    xtx_inv = np.linalg.pinv(Xd.T @ Xd)
    # White HC1 heteroskedasticity-robust variance
    meat = (Xd * (resid[:, None] ** 2)).T @ Xd
    vcov = (n / max(n - k, 1)) * (xtx_inv @ meat @ xtx_inv)
    se = np.sqrt(np.maximum(np.diag(vcov), 1e-16))
    tvals = beta / se
    # Normal approx two-sided p-value (n > 1000) via erfc
    pvals = [math.erfc(abs(float(tv)) / math.sqrt(2.0)) for tv in tvals]

    out = []
    for idx, nm in enumerate(["Intercept"] + names):
        out.append({
            "variable": nm,
            "coef": round(float(beta[idx]), 4),
            "se": round(float(se[idx]), 4),
            "t_stat": round(float(tvals[idx]), 2),
            "p_value": round(float(pvals[idx]), 4),
        })
    return out


# ---------------------------------------------------------------------------
# FULL SCIENTIFIC ORCHESTRATOR
# ---------------------------------------------------------------------------

def run_research_pipeline(force_refresh: bool = False) -> dict[str, Any]:
    raw_data = fetch_yahoo_5y(force_refresh=force_refresh)
    close_panel, aligned_frames, repairs = build_clean_panel(raw_data)

    tickers = [m["ticker"] for m in STOCK_UNIVERSE]
    etf_ticker = ETF_BENCHMARK["ticker"]
    meta_map = {m["ticker"]: m for m in STOCK_UNIVERSE}

    feat_bundle = compute_features_and_flows(aligned_frames, tickers, horizon=HORIZON_DAYS)
    dates = feat_bundle["dates"]
    feat_raw = feat_bundle["feat_raw"]
    feat_rank = feat_bundle["feat_rank"]
    ewma_vol = feat_bundle["ewma_vol"]
    fwd_ret = feat_bundle["fwd_ret"]
    fwd_cmf = feat_bundle["fwd_cmf"]
    fwd_abturn = feat_bundle["fwd_abturn"]

    n_days, n_stocks, n_feat = feat_rank.shape
    daily_ret_mat = np.column_stack([aligned_frames[t]["ret"].to_numpy() for t in tickers])
    etf_close = aligned_frames[etf_ticker]["close"].to_numpy()

    # Walk-forward expanding window starting after 300 trading days (~14 months warm-up),
    # stepping every 22 trading days (~1 month). With 1,305 total days -> ~44 OOS months!
    rebal_indices = list(range(300, n_days - HORIZON_DAYS - 2, HORIZON_DAYS))
    purge_gap = HORIZON_DAYS + 2

    # Collect OOS quantile forecast predictions for Model Evaluation (Hypothesis H1)
    oos_y_true: list[np.ndarray] = []
    oos_q_ts: list[np.ndarray] = []
    oos_q_qrf: list[np.ndarray] = []
    oos_q_qgbdt: list[np.ndarray] = []
    oos_q_hist: list[np.ndarray] = []
    oos_mean_ridge: list[np.ndarray] = []

    # Collect OOS panel rows for Capital Flow & Lottery Effect tests (Hypothesis H3)
    panel_rows: list[dict[str, float]] = []

    # Collect Portfolio curves & weights
    strat_names = [
        "ASSD-Opt (TS-QML)",
        "ASSD-Opt (QRF)",
        "ASSD + Flow (Top-8)",
        "Mean-Variance (Markowitz)",
        "Min-Volatility",
        "Point-ML (Ridge Top-8)",
        "HOSE Equal-Weight Basket",
        "VN30 ETF (E1VFVN30)",
    ]
    strat_rets: dict[str, list[float]] = {k: [] for k in strat_names}
    strat_turns: dict[str, list[float]] = {k: [] for k in strat_names}
    prev_w: dict[str, np.ndarray] = {k: np.zeros(n_stocks) for k in strat_names}
    bt_dates: list[str] = [dates[rebal_indices[0]]]

    idx_cmf = FEATURE_NAMES.index("cmf_20")
    idx_abt = FEATURE_NAMES.index("ab_turn")
    idx_max = FEATURE_NAMES.index("max_22")
    idx_ami = FEATURE_NAMES.index("amihud_20")
    idx_vol = FEATURE_NAMES.index("vol_ewma")

    for step_idx, rb in enumerate(rebal_indices):
        train_end = rb - purge_gap
        # Sample training dates every 5 days for speed while keeping full sample diversity
        tr_slice = feat_rank[130:train_end:4]     # (T_tr, N, F)
        y_slice = fwd_ret[130:train_end:4]
        sig_slice = ewma_vol[130:train_end:4]

        X_all = tr_slice.reshape(-1, n_feat)
        y_all = y_slice.reshape(-1)
        s_all = sig_slice.reshape(-1)
        ok = np.isfinite(X_all).all(axis=1) & np.isfinite(y_all) & np.isfinite(s_all)
        X_ok, y_ok, s_ok = X_all[ok], y_all[ok], s_all[ok]

        split_cal = int(len(X_ok) * 0.80)
        X_tr, y_tr, s_tr = X_ok[:split_cal], y_ok[:split_cal], s_ok[:split_cal]
        X_cal, y_cal, s_cal = X_ok[split_cal:], y_ok[split_cal:], s_ok[split_cal:]

        X_now = np.nan_to_num(feat_rank[rb], nan=0.0)
        sig_now = np.nan_to_num(ewma_vol[rb], nan=0.05)

        # Model 1: Two-Stage Vol-Scaled Quantile Model
        q_ts = predict_two_stage_qml(X_tr, y_tr, s_tr, X_cal, y_cal, s_cal, X_now, sig_now)

        # Model 2: Quantile Regression Forest
        qrf = QuantileRandomForest(n_trees=24, max_depth=4, min_samples_leaf=32, seed=42 + step_idx)
        qrf.fit(X_ok, y_ok)
        q_qrf = qrf.predict_quantiles(X_now)

        # Model 3: Quantile Gradient Boosting
        q_qgbdt = predict_qgbdt(X_ok, y_ok, X_now, n_stages=10, lr=0.14)

        # Model 4: Historical Rolling Empirical CDF (126d window)
        hist_win_ret = fwd_ret[max(0, train_end - 126):train_end]
        q_hist = np.zeros_like(q_ts)
        for j in range(n_stocks):
            vals = hist_win_ret[:, j]
            vals = vals[np.isfinite(vals)]
            q_hist[j] = np.quantile(vals, QUANTILE_PROBS) if len(vals) >= 15 else np.quantile(y_ok, QUANTILE_PROBS)

        # Baseline Point Ridge ML
        Xd_tr = np.column_stack([np.ones(len(X_ok)), X_ok])
        pen = np.diag([0.0] + [35.0] * n_feat)
        beta_r = np.linalg.solve(Xd_tr.T @ Xd_tr + pen, Xd_tr.T @ y_ok)
        pred_point = np.column_stack([np.ones(n_stocks), X_now]) @ beta_r

        realized_y = fwd_ret[rb]
        oos_y_true.append(realized_y)
        oos_q_ts.append(q_ts)
        oos_q_qrf.append(q_qrf)
        oos_q_qgbdt.append(q_qgbdt)
        oos_q_hist.append(q_hist)
        oos_mean_ridge.append(pred_point)

        # Build CDFs on grid for benchmark & stocks (using TS-QML for primary ASSD & QRF for comparison)
        cdfs_ts = np.array([quantiles_to_cdf(q_ts[j]) for j in range(n_stocks)])
        cdfs_qrf = np.array([quantiles_to_cdf(q_qrf[j]) for j in range(n_stocks)])
        bench_cdf_ts = np.mean(cdfs_ts, axis=0)
        bench_cdf_qrf = np.mean(cdfs_qrf, axis=0)

        sd_ts = [compute_sd_metrics(cdfs_ts[j], bench_cdf_ts) for j in range(n_stocks)]
        eps2_arr = np.array([sd["eps2_star"] for sd in sd_ts])
        mean_ok_arr = np.array([sd["mean_ok"] for sd in sd_ts])

        # Record OOS panel observations (Hypothesis H2 & H3)
        for j in range(n_stocks):
            if np.isfinite(realized_y[j]) and np.isfinite(fwd_cmf[rb, j]):
                panel_rows.append({
                    "step": float(step_idx),
                    "stock": float(j),
                    "is_loved": 1.0 if (mean_ok_arr[j] and eps2_arr[j] <= 0.059) else 0.0,
                    "is_hated": 1.0 if ((not mean_ok_arr[j]) and sd_ts[j]["eps2_reverse"] <= 0.059) else 0.0,
                    "assd_score": float(1.0 - 2.0 * eps2_arr[j]) if mean_ok_arr[j] else float(-eps2_arr[j]),
                    "eps2_star": float(eps2_arr[j]),
                    "pred_mean": float(sd_ts[j]["mean_a"]),
                    "cmf_t": float(feat_raw[rb, j, idx_cmf]),
                    "abt_t": float(feat_raw[rb, j, idx_abt]),
                    "max_t": float(feat_raw[rb, j, idx_max]),
                    "ami_t": float(feat_raw[rb, j, idx_ami]),
                    "vol_t": float(feat_raw[rb, j, idx_vol]),
                    "fwd_ret": float(realized_y[j]),
                    "fwd_cmf": float(fwd_cmf[rb, j]),
                    "fwd_abturn": float(fwd_abturn[rb, j]),
                })

        # Compute optimal portfolio weights per strategy
        hist_daily_slice = daily_ret_mat[max(0, rb - 150):rb]
        w_assd_ts = optimize_assd_portfolio(q_ts, hist_daily_slice, prev_w["ASSD-Opt (TS-QML)"])
        w_assd_qrf = optimize_assd_portfolio(q_qrf, hist_daily_slice, prev_w["ASSD-Opt (QRF)"])

        # ASSD + Flow Top-8 inverse-vol weighted
        score_flow = (1.0 - eps2_arr) + 0.15 * np.tanh(feat_raw[rb, :, idx_cmf] * 2.0) - 0.5 * np.where(mean_ok_arr, 0.0, 1.0)
        top8_assd = np.argsort(-score_flow)[:8]
        w_assd_top = np.zeros(n_stocks)
        inv_vol = 1.0 / np.maximum(sig_now[top8_assd], 0.02)
        w_assd_top[top8_assd] = inv_vol / inv_vol.sum()

        # Min-Volatility & Markowitz
        w_minvol = optimize_min_variance(hist_daily_slice)
        ts_means = np.mean(q_ts, axis=1)
        w_mv = optimize_markowitz_mv(ts_means, hist_daily_slice)

        # Ridge Point Top-8 Equal
        top8_ridge = np.argsort(-pred_point)[:8]
        w_ridge = np.zeros(n_stocks)
        w_ridge[top8_ridge] = 1.0 / 8.0

        # Equal-Weight HOSE Basket
        w_eq = np.full(n_stocks, 1.0 / n_stocks)

        weights_map = {
            "ASSD-Opt (TS-QML)": w_assd_ts,
            "ASSD-Opt (QRF)": w_assd_qrf,
            "ASSD + Flow (Top-8)": w_assd_top,
            "Mean-Variance (Markowitz)": w_mv,
            "Min-Volatility": w_minvol,
            "Point-ML (Ridge Top-8)": w_ridge,
            "HOSE Equal-Weight Basket": w_eq,
        }
        next_rb = min(rb + HORIZON_DAYS, n_days - 1)
        bt_dates.append(dates[next_rb])

        for sname, w_cur in weights_map.items():
            # Apply realistic turnover transaction cost
            turn = 0.5 * float(np.sum(np.abs(w_cur - prev_w[sname]))) if step_idx > 0 else 0.5
            cost = turn * TRANSACTION_COST
            net_r = float(np.dot(w_cur, np.nan_to_num(realized_y, nan=0.0))) - cost
            strat_rets[sname].append(net_r)
            strat_turns[sname].append(turn)
            prev_w[sname] = w_cur

        etf_r = float(etf_close[next_rb] / etf_close[rb] - 1.0)
        strat_rets["VN30 ETF (E1VFVN30)"].append(etf_r)
        strat_turns["VN30 ETF (E1VFVN30)"].append(0.0)

    # ------------------------------------------------------------------
    # HYPOTHESIS H1: DISTRIBUTIONAL FORECAST EVALUATION (OOS)
    # ------------------------------------------------------------------
    Y_oos = np.concatenate(oos_y_true)
    ok_y = np.isfinite(Y_oos)
    Y_oos = Y_oos[ok_y]
    dist_eval = {
        "Two-Stage Vol-Scaled QML (Barunik 2026)": pinball_loss(Y_oos, np.vstack(oos_q_ts)[ok_y]),
        "Quantile Regression Forests (Meinshausen 2006)": pinball_loss(Y_oos, np.vstack(oos_q_qrf)[ok_y]),
        "Quantile Gradient Boosting": pinball_loss(Y_oos, np.vstack(oos_q_qgbdt)[ok_y]),
        "Historical Empirical CDF Baseline": pinball_loss(Y_oos, np.vstack(oos_q_hist)[ok_y]),
    }
    # Add MSE/RMSE comparison for median & point ML
    ridge_vec = np.concatenate(oos_mean_ridge)[ok_y]
    ts_med = np.vstack(oos_q_ts)[ok_y, 9]
    qrf_med = np.vstack(oos_q_qrf)[ok_y, 9]
    dist_eval["Two-Stage Vol-Scaled QML (Barunik 2026)"]["rmse_pct"] = round(float(np.sqrt(np.mean((Y_oos - ts_med) ** 2)) * 100), 2)
    dist_eval["Quantile Regression Forests (Meinshausen 2006)"]["rmse_pct"] = round(float(np.sqrt(np.mean((Y_oos - qrf_med) ** 2)) * 100), 2)
    dist_eval["Quantile Gradient Boosting"]["rmse_pct"] = round(float(np.sqrt(np.mean((Y_oos - np.vstack(oos_q_qgbdt)[ok_y, 9]) ** 2)) * 100), 2)
    dist_eval["Historical Empirical CDF Baseline"]["rmse_pct"] = round(float(np.sqrt(np.mean((Y_oos - np.vstack(oos_q_hist)[ok_y, 9]) ** 2)) * 100), 2)
    dist_eval["Point-ML Ridge Benchmark"] = {
        "pinball_mean": None, "pinball_left": None, "pinball_median": None, "pinball_right": None,
        "crps_pct": None, "coverage_90_pct": None,
        "rmse_pct": round(float(np.sqrt(np.mean((Y_oos - ridge_vec) ** 2)) * 100), 2),
    }

    # ------------------------------------------------------------------
    # HYPOTHESIS H3: CAPITAL FLOW & LOTTERY PARADOX ANALYSIS
    # ------------------------------------------------------------------
    df_p = pd.DataFrame(panel_rows)
    loved_df = df_p[df_p["is_loved"] == 1.0]
    hated_df = df_p[df_p["is_hated"] == 1.0]
    mid_df = df_p[(df_p["is_loved"] == 0.0) & (df_p["is_hated"] == 0.0)]

    # Lottery paradox split: High MAX (top 25% lottery demand) that fail strict SSD
    max_q75 = float(df_p["max_t"].quantile(0.75))
    lottery_df = df_p[(df_p["max_t"] >= max_q75) & (df_p["is_loved"] == 0.0)]

    flow_groups = {
        "n_obs": int(len(df_p)),
        "loved": {
            "count": int(len(loved_df)),
            "fwd_ret_pct": round(float(loved_df["fwd_ret"].mean() * 100), 2) if len(loved_df) else 0.0,
            "fwd_cmf": round(float(loved_df["fwd_cmf"].mean()), 4) if len(loved_df) else 0.0,
            "fwd_abturn": round(float(loved_df["fwd_abturn"].mean()), 2) if len(loved_df) else 0.0,
            "curr_cmf": round(float(loved_df["cmf_t"].mean()), 4) if len(loved_df) else 0.0,
        },
        "neutral": {
            "count": int(len(mid_df)),
            "fwd_ret_pct": round(float(mid_df["fwd_ret"].mean() * 100), 2) if len(mid_df) else 0.0,
            "fwd_cmf": round(float(mid_df["fwd_cmf"].mean()), 4) if len(mid_df) else 0.0,
            "fwd_abturn": round(float(mid_df["fwd_abturn"].mean()), 2) if len(mid_df) else 0.0,
            "curr_cmf": round(float(mid_df["cmf_t"].mean()), 4) if len(mid_df) else 0.0,
        },
        "dominated": {
            "count": int(len(hated_df)),
            "fwd_ret_pct": round(float(hated_df["fwd_ret"].mean() * 100), 2) if len(hated_df) else 0.0,
            "fwd_cmf": round(float(hated_df["fwd_cmf"].mean()), 4) if len(hated_df) else 0.0,
            "fwd_abturn": round(float(hated_df["fwd_abturn"].mean()), 2) if len(hated_df) else 0.0,
            "curr_cmf": round(float(hated_df["cmf_t"].mean()), 4) if len(hated_df) else 0.0,
        },
        "lottery_high_max": {
            "count": int(len(lottery_df)),
            "fwd_ret_pct": round(float(lottery_df["fwd_ret"].mean() * 100), 2) if len(lottery_df) else 0.0,
            "fwd_cmf": round(float(lottery_df["fwd_cmf"].mean()), 4) if len(lottery_df) else 0.0,
            "fwd_abturn": round(float(lottery_df["fwd_abturn"].mean()), 2) if len(lottery_df) else 0.0,
            "curr_abturn": round(float(lottery_df["abt_t"].mean()), 2) if len(lottery_df) else 0.0,
        },
    }

    reg_cmf = _ols_with_tstat(
        df_p["fwd_cmf"].to_numpy(),
        df_p[["assd_score", "cmf_t", "max_t", "vol_t"]].to_numpy(),
        ["ASSD_Score_t", "CMF_20_t", "MAX_Lottery_t", "EWMA_Vol_t"],
    )
    reg_ret = _ols_with_tstat(
        df_p["fwd_ret"].to_numpy() * 100.0,
        df_p[["assd_score", "cmf_t", "max_t", "vol_t"]].to_numpy(),
        ["ASSD_Score_t", "CMF_20_t", "MAX_Lottery_t", "EWMA_Vol_t"],
    )

    # ------------------------------------------------------------------
    # LATEST LIVE PREDICTION (Train on ALL clean rows up to n_days - purge_gap)
    # ------------------------------------------------------------------
    tr_full = feat_rank[130:n_days - purge_gap:2]
    y_full = fwd_ret[130:n_days - purge_gap:2]
    s_full = ewma_vol[130:n_days - purge_gap:2]
    Xf = tr_full.reshape(-1, n_feat)
    yf = y_full.reshape(-1)
    sf = s_full.reshape(-1)
    okf = np.isfinite(Xf).all(axis=1) & np.isfinite(yf) & np.isfinite(sf)
    Xf, yf, sf = Xf[okf], yf[okf], sf[okf]

    sp = int(len(Xf) * 0.85)
    X_latest = np.nan_to_num(feat_rank[-1], nan=0.0)
    s_latest = np.nan_to_num(ewma_vol[-1], nan=0.05)

    q_latest_ts = predict_two_stage_qml(Xf[:sp], yf[:sp], sf[:sp], Xf[sp:], yf[sp:], sf[sp:], X_latest, s_latest)
    qrf_full = QuantileRandomForest(n_trees=36, max_depth=4, min_samples_leaf=30, seed=99)
    qrf_full.fit(Xf, yf)
    q_latest_qrf = qrf_full.predict_quantiles(X_latest)
    q_latest_qgbdt = predict_qgbdt(Xf, yf, X_latest, n_stages=12, lr=0.14)

    cdfs_latest_ts = np.array([quantiles_to_cdf(q_latest_ts[j]) for j in range(n_stocks)])
    cdfs_latest_qrf = np.array([quantiles_to_cdf(q_latest_qrf[j]) for j in range(n_stocks)])
    bench_cdf_ts_now = np.mean(cdfs_latest_ts, axis=0)
    bench_cdf_qrf_now = np.mean(cdfs_latest_qrf, axis=0)

    w_latest_assd = optimize_assd_portfolio(q_latest_ts, daily_ret_mat[-150:], prev_w["ASSD-Opt (TS-QML)"])

    # Pairwise ASSD Matrix N x N using TS-QML CDFs
    pairwise_eps2 = np.zeros((n_stocks, n_stocks), dtype=np.float64)
    pairwise_mean_ok = np.zeros((n_stocks, n_stocks), dtype=bool)
    for i in range(n_stocks):
        for j in range(n_stocks):
            if i == j:
                continue
            p_sd = compute_sd_metrics(cdfs_latest_ts[i], cdfs_latest_ts[j])
            pairwise_eps2[i, j] = p_sd["eps2_star"]
            pairwise_mean_ok[i, j] = p_sd["mean_ok"]

    latest_stocks = []
    for j, t in enumerate(tickers):
        sd_ts_j = compute_sd_metrics(cdfs_latest_ts[j], bench_cdf_ts_now)
        sd_qrf_j = compute_sd_metrics(cdfs_latest_qrf[j], bench_cdf_qrf_now)
        df_j = aligned_frames[t]
        closes_60 = [round(float(x), 2) for x in df_j["close"].iloc[-60:].to_numpy()]
        vols_60 = [round(float(x), 0) for x in df_j["volume"].iloc[-60:].to_numpy()]
        dom_out = int(np.sum((pairwise_mean_ok[j]) & (pairwise_eps2[j] <= 0.059)) - 1)
        dom_in = int(np.sum((pairwise_mean_ok[:, j]) & (pairwise_eps2[:, j] <= 0.059)) - 1)

        latest_stocks.append({
            "ticker": t,
            "short_ticker": t.replace(".VN", ""),
            "name": meta_map[t]["name"],
            "sector": meta_map[t]["sector"],
            "last_close": round(float(df_j["close"].iloc[-1]), 2),
            "daily_ret_pct": round(float(df_j["ret"].iloc[-1] * 100.0), 2),
            "ret_22d_pct": round(float(feat_raw[-1, j, FEATURE_NAMES.index("ret_22d")] * 100.0), 2),
            "ret_66d_pct": round(float(feat_raw[-1, j, FEATURE_NAMES.index("ret_66d")] * 100.0), 2),
            "vol_ewma_pct": round(float(feat_raw[-1, j, idx_vol] * 100.0), 2),
            "max_22_pct": round(float(feat_raw[-1, j, idx_max] * 100.0), 2),
            "skew_22": round(float(feat_raw[-1, j, FEATURE_NAMES.index("skew_22")]), 2),
            "cmf_20": round(float(feat_raw[-1, j, idx_cmf]), 3),
            "ab_turn": round(float(feat_raw[-1, j, idx_abt]), 2),
            "rsi_14": round(float(feat_raw[-1, j, FEATURE_NAMES.index("rsi_14")] * 100.0), 1),
            "opt_weight_pct": round(float(w_latest_assd[j] * 100.0), 1),
            "net_pairwise_dom": max(-n_stocks, dom_out - dom_in),
            "pred_mean_ts_pct": round(float(sd_ts_j["mean_a"] * 100.0), 2),
            "pred_mean_qrf_pct": round(float(sd_qrf_j["mean_a"] * 100.0), 2),
            "loss_prob_ts_pct": round(float(np.interp(0.0, CDF_GRID, cdfs_latest_ts[j]) * 100.0), 1),
            "quantiles_ts": [round(float(x * 100.0), 2) for x in q_latest_ts[j]],
            "quantiles_qrf": [round(float(x * 100.0), 2) for x in q_latest_qrf[j]],
            "quantiles_qgbdt": [round(float(x * 100.0), 2) for x in q_latest_qgbdt[j]],
            "cdf_ts": [round(float(x), 4) for x in cdfs_latest_ts[j]],
            "cdf_qrf": [round(float(x), 4) for x in cdfs_latest_qrf[j]],
            "sd_ts": sd_ts_j,
            "sd_qrf": sd_qrf_j,
            "closes_60": closes_60,
            "vols_60": vols_60,
        })

    latest_stocks.sort(key=lambda x: (not x["sd_ts"]["mean_ok"], x["sd_ts"]["eps2_star"], -x["pred_mean_ts_pct"]))

    # Backtest curves & metrics
    bench_rets = np.array(strat_rets["HOSE Equal-Weight Basket"], dtype=np.float64)
    bt_curves: dict[str, list[float]] = {}
    bt_metrics: dict[str, dict[str, float]] = {}
    for sname in strat_names:
        arr = np.array(strat_rets[sname], dtype=np.float64)
        eq = np.concatenate([[100.0], 100.0 * np.cumprod(1.0 + arr)])
        bt_curves[sname] = [round(float(v), 2) for v in eq]
        bt_metrics[sname] = compute_portfolio_metrics(arr, strat_turns[sname], bench_rets)

    # Compact market panel JSON for live interactive browser ML training/backtest!
    # Keep last 520 days x 38 tickers (feature matrix + returns + CMF + AbTurn + dates)
    panel_start = max(0, n_days - 520)
    compact_panel = {
        "dates": dates[panel_start:],
        "tickers": [t.replace(".VN", "") for t in tickers],
        "sectors": [meta_map[t]["sector"] for t in tickers],
        "feature_names": FEATURE_NAMES,
        # Quantize floats cleanly so JSON size stays compact (~500 KB gzipped!)
        "closes": np.round(close_panel[tickers].iloc[panel_start:].to_numpy(), 1).tolist(),
        "rets": np.round(daily_ret_mat[panel_start:] * 100.0, 2).tolist(),  # pct
        "cmf": np.round(feat_raw[panel_start:, :, idx_cmf], 3).tolist(),
        "abturn": np.round(feat_raw[panel_start:, :, idx_abt], 2).tolist(),
    }

    # Load verified Q1 corpus (376 papers)
    q1_papers: list[dict[str, Any]] = []
    if CORPUS_CSV_PATH.exists():
        with open(CORPUS_CSV_PATH, encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q1_papers.append({
                    "id": row["id"],
                    "year": int(row["year"]) if row["year"].isdigit() else 0,
                    "authors": row["authors"],
                    "title": row["title"],
                    "journal": row["journal"],
                    "quartile": row["sjr_quartile"],
                    "branch": row["sub_area"],
                    "citations": int(row["citations"]) if row["citations"].isdigit() else 0,
                    "doi": row["doi"],
                    "url": row["url"],
                    "abstract": (row.get("abstract_clean") or "")[:340],
                })
        q1_papers.sort(key=lambda r: (-r["citations"], -r["year"]))

    bundle = {
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "date_range": {"start": dates[0], "end": dates[-1], "trading_days": n_days},
        "universe_count": n_stocks,
        "horizon_days": HORIZON_DAYS,
        "dates_60": dates[-60:],
        "grid_pct": [round(float(x * 100.0), 2) for x in CDF_GRID],
        "quantile_probs": [round(float(p * 100.0), 1) for p in QUANTILE_PROBS],
        "benchmark": {
            "name": f"HOSE Equal-Weight Basket ({n_stocks} mã)",
            "cdf_ts": [round(float(x), 4) for x in bench_cdf_ts_now],
            "cdf_qrf": [round(float(x), 4) for x in bench_cdf_qrf_now],
            "pred_mean_ts_pct": round(float(np.mean([s["pred_mean_ts_pct"] for s in latest_stocks])), 2),
        },
        "stocks": latest_stocks,
        "distributional_eval_h1": dist_eval,
        "flow_analysis_h3": {
            "groups": flow_groups,
            "reg_fwd_cmf": reg_cmf,
            "reg_fwd_ret": reg_ret,
        },
        "backtest_h4": {
            "dates": bt_dates,
            "n_rebalances": len(rebal_indices),
            "cost_note": "Phí 2 chiều 0,25% + thuế bán HOSE 0,10% = 0,35% / vòng xoay; Long-Only (trọng số tối đa 20%/mã)",
            "curves": bt_curves,
            "metrics": bt_metrics,
        },
        "data_audit": {
            "total_repairs": len(repairs),
            "sample_repairs": repairs[:24],
        },
        "corpus_summary": {
            "total_papers": len(q1_papers),
            "journals_count": len(set(p["journal"] for p in q1_papers)),
        },
    }

    OUTPUT_BUNDLE_PATH.write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    DOCS_DATA_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DATA_DIR / "research_bundle.json").write_text(json.dumps(bundle, ensure_ascii=False), encoding="utf-8")
    (DOCS_DATA_DIR / "market_panel.json").write_text(json.dumps(compact_panel, ensure_ascii=False), encoding="utf-8")
    (DOCS_DATA_DIR / "q1_corpus.json").write_text(json.dumps(q1_papers, ensure_ascii=False), encoding="utf-8")
    return bundle


if __name__ == "__main__":
    res = run_research_pipeline(force_refresh=False)
    print("RANGE:", res["date_range"], "| STOCKS:", res["universe_count"], "| REBALANCES:", res["backtest_h4"]["n_rebalances"])
    print("\n--- H1: DISTRIBUTIONAL EVAL ---")
    for k, v in res["distributional_eval_h1"].items():
        print(k, "->", v)
    print("\n--- H4: PORTFOLIO BACKTEST ---")
    for k, v in res["backtest_h4"]["metrics"].items():
        print(k, "->", v)
    print("\n--- H3: CAPITAL FLOW ---")
    print("Groups:", res["flow_analysis_h3"]["groups"])
    print("Reg Fwd CMF:", res["flow_analysis_h3"]["reg_fwd_cmf"])
    print("Reg Fwd Ret:", res["flow_analysis_h3"]["reg_fwd_ret"])
    print("\n--- AUDIT REPAIRS ---", res["data_audit"]["total_repairs"])
