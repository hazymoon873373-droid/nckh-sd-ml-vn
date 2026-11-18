/**
 * app.js — Main Interactive Terminal Orchestrator for VN-QUANT NCKH Web Workbench
 * Connects pre-computed 5y Python research results with live In-Browser ML,
 * interactive SVG crosshair charts, pairwise ASSD testing, and 376-paper Q1 explorer.
 */
(() => {
  "use strict";

  let BUNDLE = null;
  let PANEL = null;
  let CORPUS = null;

  let activeModel = "ts_qml"; // ts_qml | qrf | qgbdt
  let currentEps = 0.059;
  let selectedTicker = "VCB.VN";
  let compareTarget = "BENCH";
  let sortCol = "eps2";
  let sortAsc = true;
  let liveMlRanOnce = false;

  const qs = (s) => document.querySelector(s);
  const qsa = (s) => Array.from(document.querySelectorAll(s));

  const fmtPct = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "--" : `${Number(v).toFixed(d).replace(".", ",")}%`);
  const fmtNum = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "--" : Number(v).toFixed(d).replace(".", ","));

  function showStatus(msg, isError = false) {
    const el = qs("#global-status");
    el.textContent = msg;
    el.classList.remove("is-hidden", "is-error");
    if (isError) el.classList.add("is-error");
  }

  function hideStatus() {
    qs("#global-status").classList.add("is-hidden");
  }

  function getStockView(s) {
    if (activeModel === "qrf") {
      return {
        meanPct: s.pred_mean_qrf_pct,
        quantiles: s.quantiles_qrf,
        cdf: s.cdf_qrf,
        sd: s.sd_qrf,
        benchCdf: BUNDLE.benchmark.cdf_qrf,
      };
    }
    if (activeModel === "qgbdt") {
      if (!s._cached_qgbdt_cdf) {
        const taus = BUNDLE.quantile_probs.map((p) => p / 100.0);
        s._cached_qgbdt_cdf = window.SDMath.quantilesToCdf(s.quantiles_qgbdt, taus, BUNDLE.grid_pct);
      }
      if (!BUNDLE.benchmark._cached_qgbdt_cdf) {
        const n = BUNDLE.stocks.length;
        const gLen = BUNDLE.grid_pct.length;
        const avg = new Array(gLen).fill(0);
        for (const st of BUNDLE.stocks) {
          if (!st._cached_qgbdt_cdf) {
            const taus = BUNDLE.quantile_probs.map((p) => p / 100.0);
            st._cached_qgbdt_cdf = window.SDMath.quantilesToCdf(st.quantiles_qgbdt, taus, BUNDLE.grid_pct);
          }
          for (let i = 0; i < gLen; i++) avg[i] += st._cached_qgbdt_cdf[i] / n;
        }
        BUNDLE.benchmark._cached_qgbdt_cdf = avg;
      }
      if (!s._cached_qgbdt_sd) {
        s._cached_qgbdt_sd = window.SDMath.computeSdMetrics(s._cached_qgbdt_cdf, BUNDLE.benchmark._cached_qgbdt_cdf, BUNDLE.grid_pct);
      }
      return {
        meanPct: s._cached_qgbdt_sd.mean_a * 100.0,
        quantiles: s.quantiles_qgbdt,
        cdf: s._cached_qgbdt_cdf,
        sd: s._cached_qgbdt_sd,
        benchCdf: BUNDLE.benchmark._cached_qgbdt_cdf,
      };
    }
    return {
      meanPct: s.pred_mean_ts_pct,
      quantiles: s.quantiles_ts,
      cdf: s.cdf_ts,
      sd: s.sd_ts,
      benchCdf: BUNDLE.benchmark.cdf_ts,
    };
  }

  function renderTickerRibbon() {
    const track = qs("#ticker-track");
    const itemsHtml = BUNDLE.stocks.map((s) => {
      const sv = getStockView(s);
      const cls = window.SDMath.classifySd(sv.sd, currentEps);
      const chgCls = s.daily_ret_pct >= 0 ? "pos" : "neg";
      const sign = s.daily_ret_pct >= 0 ? "+" : "";
      const tag = cls.isLoved ? "YÊU" : cls.cls === "badge-bear" ? "BỊ CHI PHỐI" : "TRUNG";
      const tagCls = cls.isLoved ? "pos" : cls.cls === "badge-bear" ? "neg" : "warn";
      return `<button type="button" class="tick-chip" data-pick="${s.ticker}">
        <span class="tick-sym">${s.short_ticker}</span>
        <span class="tick-px">${fmtNum(s.last_close, 1)}</span>
        <span class="${chgCls}">${sign}${fmtPct(s.daily_ret_pct, 1)}</span>
        <span class="${tagCls}">${tag} (${fmtPct(sv.sd.eps2_star * 100, 1)})</span>
      </button>`;
    }).join("");
    // Duplicate track seamlessly for continuous horizontal marquee
    track.innerHTML = itemsHtml + itemsHtml;
    qsa(".tick-chip").forEach((btn) => {
      btn.addEventListener("click", () => {
        selectedTicker = btn.dataset.pick;
        activateTab("terminal");
        renderScreener();
        renderDetailTerminal();
      });
    });
  }

  function renderKPIs() {
    const stocks = BUNDLE.stocks;
    let loveCount = 0;
    const secGroups = {};
    for (const s of stocks) {
      const sv = getStockView(s);
      const cls = window.SDMath.classifySd(sv.sd, currentEps);
      if (cls.isLoved) loveCount++;
      if (!secGroups[s.sector]) secGroups[s.sector] = { total: 0, loved: 0, epsSum: 0 };
      secGroups[s.sector].total++;
      if (cls.isLoved) secGroups[s.sector].loved++;
      secGroups[s.sector].epsSum += sv.sd.eps2_star * 100;
    }

    qs("#kpi-loved").textContent = `${loveCount} / ${stocks.length}`;
    qs("#kpi-loved-sub").textContent = `${fmtPct((100 * loveCount) / stocks.length, 1)} rổ mã ở ngưỡng ε₂ = ${fmtPct(currentEps * 100, 1)}`;

    const topSec = Object.entries(secGroups)
      .map(([sec, d]) => ({ sec, ratio: (100 * d.loved) / d.total, loved: d.loved, total: d.total, avgEps: d.epsSum / d.total }))
      .sort((a, b) => b.ratio - a.ratio || a.avgEps - b.avgEps)[0];
    qs("#kpi-top-sector").textContent = topSec.sec;
    qs("#kpi-top-sector-sub").textContent = `${topSec.loved}/${topSec.total} mã "Yêu" (${fmtPct(topSec.ratio, 0)}) | ε₂* TB ${fmtPct(topSec.avgEps, 2)}`;

    const mAssd = BUNDLE.backtest_h4.metrics["ASSD-Opt (TS-QML)"];
    const mBasket = BUNDLE.backtest_h4.metrics["HOSE Equal-Weight Basket"];
    qs("#kpi-oos-ret").textContent = `+${fmtPct(mAssd.cum_return_pct, 2)}`;
    qs("#kpi-oos-sub").textContent = `Basket: +${fmtPct(mBasket.cum_return_pct, 1)} | ETF: +${fmtPct(BUNDLE.backtest_h4.metrics["VN30 ETF (E1VFVN30)"].cum_return_pct, 1)} (${BUNDLE.backtest_h4.n_rebalances} kỳ)`;

    qs("#kpi-oos-risk").textContent = `${fmtNum(mAssd.sharpe)} / ${fmtNum(mAssd.sortino)} / ${fmtPct(mAssd.max_drawdown_pct, 1)}`;
    qs("#kpi-oos-risk-sub").textContent = `Omega ${fmtNum(mAssd.omega)} | Calmar ${fmtNum(mAssd.calmar)} (sau phí 0,35%)`;

    const assdReg = BUNDLE.flow_analysis_h3.reg_fwd_cmf.find((r) => r.variable === "ASSD_Score_t");
    const maxReg = BUNDLE.flow_analysis_h3.reg_fwd_cmf.find((r) => r.variable === "MAX_Lottery_t");
    if (assdReg && maxReg) {
      qs("#kpi-flow-pval").textContent = `p = ${fmtNum(assdReg.p_value, 4)} (t=${fmtNum(assdReg.t_stat)})`;
      qs("#kpi-flow-sub").textContent = `Bẫy vé số MAX(t): t=${fmtNum(maxReg.t_stat)} (p=${fmtNum(maxReg.p_value, 4)})`;
    }
  }

  function populateDropdowns() {
    const sectors = Array.from(new Set(BUNDLE.stocks.map((s) => s.sector))).sort();
    const secSel = qs("#sector-filter");
    sectors.forEach((sec) => {
      const opt = document.createElement("option");
      opt.value = sec;
      opt.textContent = sec;
      secSel.appendChild(opt);
    });

    const compSel = qs("#compare-target");
    BUNDLE.stocks.forEach((s) => {
      const opt = document.createElement("option");
      opt.value = s.ticker;
      opt.textContent = `${s.short_ticker} — ${s.name}`;
      compSel.appendChild(opt);
    });
  }

  function renderScreener() {
    const qText = qs("#search-ticker").value.trim().toLowerCase();
    const secVal = qs("#sector-filter").value;
    const verdVal = qs("#verdict-filter").value;

    let rows = BUNDLE.stocks.map((s) => {
      const sv = getStockView(s);
      const cls = window.SDMath.classifySd(sv.sd, currentEps);
      return { s, sv, cls };
    });

    rows = rows.filter(({ s, cls }) => {
      if (qText && !s.short_ticker.toLowerCase().includes(qText) && !s.name.toLowerCase().includes(qText)) return false;
      if (secVal !== "ALL" && s.sector !== secVal) return false;
      if (verdVal === "LOVE" && !cls.isLoved) return false;
      if (verdVal === "NEUTRAL" && cls.cls !== "badge-neutral") return false;
      if (verdVal === "HATE" && cls.cls !== "badge-bear") return false;
      return true;
    });

    const getVal = (item) => {
      switch (sortCol) {
        case "ticker": return item.s.short_ticker;
        case "price": return item.s.last_close;
        case "mean": return item.sv.meanPct;
        case "eps2": return item.sv.sd.eps2_star;
        case "cmf": return item.s.cmf_20;
        case "weight": return item.s.opt_weight_pct;
        default: return item.sv.sd.eps2_star;
      }
    };

    rows.sort((a, b) => {
      const va = getVal(a);
      const vb = getVal(b);
      if (typeof va === "string") return sortAsc ? va.localeCompare(vb) : vb.localeCompare(va);
      return sortAsc ? va - vb : vb - va;
    });

    qs("#screener-count").textContent = `${rows.length} / ${BUNDLE.stocks.length} mã`;
    const tbody = qs("#screener-body");
    tbody.innerHTML = "";
    for (const { s, sv, cls } of rows) {
      const tr = document.createElement("tr");
      tr.className = "click-row" + (s.ticker === selectedTicker ? " is-selected" : "");
      tr.tabIndex = 0;
      tr.setAttribute("role", "button");
      const lotNote = s.max_22_pct >= 5.8 ? ' <span class="badge badge-neutral" title="Rủi ro vé số MAX cao">MAX</span>' : "";
      const shortBadge = cls.label.includes("SSD Chuẩn") ? "YÊU · SSD" : cls.label.includes("Almost SD") ? "YÊU · ASSD" : cls.cls === "badge-bear" ? "KHÔNG YÊU" : "TRUNG TÍNH";
      tr.innerHTML = `
        <td><strong>${s.short_ticker}</strong>${lotNote}<br><span class="panel-sub">${s.sector}</span></td>
        <td class="num">${fmtNum(s.last_close, 1)}<br><span class="panel-sub ${s.daily_ret_pct >= 0 ? "pos" : "neg"}">${s.daily_ret_pct >= 0 ? "+" : ""}${fmtPct(s.daily_ret_pct, 1)}</span></td>
        <td class="num ${sv.meanPct >= 0 ? "pos" : "neg"}">${fmtPct(sv.meanPct, 2)}</td>
        <td class="num">${fmtPct(sv.sd.eps2_star * 100, 2)}</td>
        <td class="num ${s.cmf_20 >= 0 ? "pos" : "neg"}">${fmtNum(s.cmf_20, 2)}</td>
        <td class="num">${fmtPct(s.opt_weight_pct, 1)}</td>
        <td><span class="badge ${cls.cls}">${shortBadge}</span></td>
      `;
      tr.addEventListener("click", () => {
        selectedTicker = s.ticker;
        renderScreener();
        renderDetailTerminal();
      });
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectedTicker = s.ticker;
          renderScreener();
          renderDetailTerminal();
        }
      });
      tbody.appendChild(tr);
    }
  }

  function renderDetailTerminal() {
    const s = BUNDLE.stocks.find((x) => x.ticker === selectedTicker) || BUNDLE.stocks[0];
    selectedTicker = s.ticker;
    const sv = getStockView(s);

    let cdfB = sv.benchCdf;
    let labelB = "VN-Basket";
    let sdPair = sv.sd;
    if (compareTarget !== "BENCH" && compareTarget !== s.ticker) {
      const peer = BUNDLE.stocks.find((x) => x.ticker === compareTarget);
      if (peer) {
        const pv = getStockView(peer);
        cdfB = pv.cdf;
        labelB = peer.short_ticker;
        sdPair = window.SDMath.computeSdMetrics(sv.cdf, cdfB, BUNDLE.grid_pct);
      }
    }

    const cls = window.SDMath.classifySd(sdPair, currentEps);
    qs("#detail-title").textContent = `${s.short_ticker} — ${s.name} (${s.sector})`;
    qs("#detail-subtitle").textContent = `So sánh Phân phối ML kỳ 22 phiên: ${s.short_ticker} vs ${labelB}`;
    const badge = qs("#detail-badge");
    badge.className = `badge ${cls.cls}`;
    badge.textContent = cls.label;

    const idxZero = BUNDLE.grid_pct.findIndex((g) => g >= 0);
    const liveLossProb = idxZero >= 0 ? sv.cdf[idxZero] * 100.0 : s.loss_prob_ts_pct;
    qs("#q-px").textContent = `${fmtNum(s.last_close, 1)} (${s.daily_ret_pct >= 0 ? "+" : ""}${fmtPct(s.daily_ret_pct, 2)})`;
    qs("#q-mom").textContent = `${fmtPct(s.ret_22d_pct, 1)} / ${fmtPct(s.ret_66d_pct, 1)}`;
    qs("#q-mean").textContent = `${fmtPct(sv.meanPct, 2)} (Chuẩn: ${fmtPct(sdPair.mean_b * 100, 2)})`;
    qs("#q-risk").textContent = `Vol ${fmtPct(s.vol_ewma_pct, 1)} | P(<0)=${fmtPct(liveLossProb, 0)}`;
    qs("#q-flow").textContent = `CMF ${fmtNum(s.cmf_20, 2)} | ${fmtNum(s.ab_turn, 2)}x`;
    qs("#q-opt").textContent = `${fmtPct(s.opt_weight_pct, 1)} (Điểm cặp ${s.net_pairwise_dom >= 0 ? "+" : ""}${s.net_pairwise_dom})`;

    window.ChartEngine.renderFanChart(
      qs("#svg-fan"), qs("#tip-fan"), s.closes_60, BUNDLE.dates_60, sv.quantiles, BUNDLE.horizon_days, s.short_ticker
    );
    window.ChartEngine.renderDualCdfCharts(
      qs("#svg-cdf"), qs("#tip-cdf"), qs("#svg-f2"), qs("#tip-f2"),
      BUNDLE.grid_pct, sv.cdf, cdfB, sdPair.f2_a, sdPair.f2_b, s.short_ticker, labelB
    );

    qs("#assd-formula-panel").innerHTML = `
      <div><strong>Kết quả Kiểm định Almost Stochastic Dominance trực tiếp (${s.short_ticker} vs ${labelB}):</strong></div>
      <div class="formula-mono">
        1) Điều kiện Kỳ vọng Tzeng et al. (2013): E[R<sub>${s.short_ticker}</sub>] = <strong>${fmtPct(sdPair.mean_a * 100, 2)}</strong> vs E[R<sub>${labelB}</sub>] = ${fmtPct(sdPair.mean_b * 100, 2)} &rarr; <span class="${sdPair.mean_ok ? "pos" : "neg"}"><strong>${sdPair.mean_ok ? "ĐẠT" : "KHÔNG ĐẠT"}</strong></span><br/>
        2) Tỷ lệ diện tích vi phạm ASSD bậc 2: &epsilon;<sub>2</sub>* = ${sdPair.area_viol2.toFixed(6)} / ${sdPair.area_total2.toFixed(6)} = <strong>${fmtPct(sdPair.eps2_star * 100, 2)}</strong> (Ngưỡng đang xét: ${fmtPct(currentEps * 100, 1)})<br/>
        3) Tỷ lệ vi phạm FSD (&epsilon;<sub>1</sub>*): ${fmtPct(sdPair.eps1_star * 100, 2)} | Điểm MAX-22 (Vé số): ${fmtPct(s.max_22_pct, 2)} | RSI-14: ${fmtNum(s.rsi_14, 1)}
      </div>
    `;
  }

  function renderH1AndFlowsAndBacktest() {
    // Tab 2 H1 Eval Table
    const h1Body = qs("#h1-eval-body");
    h1Body.innerHTML = "";
    Object.entries(BUNDLE.distributional_eval_h1).forEach(([name, m]) => {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${name}</strong></td>
        <td class="num">${m.pinball_mean !== null ? fmtPct(m.pinball_mean, 4) : "--"}</td>
        <td class="num">${m.pinball_left !== null ? fmtPct(m.pinball_left, 4) : "--"}</td>
        <td class="num">${m.pinball_median !== null ? fmtPct(m.pinball_median, 4) : "--"}</td>
        <td class="num">${m.crps_pct !== null ? fmtPct(m.crps_pct, 4) : "--"}</td>
        <td class="num">${m.coverage_90_pct !== null ? fmtPct(m.coverage_90_pct, 2) : "--"}</td>
        <td class="num">${fmtPct(m.rmse_pct, 2)}</td>
      `;
      h1Body.appendChild(tr);
    });

    // Tab 3 Sector Flows & H3
    renderSectorFlows();
    const g = BUNDLE.flow_analysis_h3.groups;
    qs("#h3-obs-count").textContent = `${g.n_obs} quan sát cổ phiếu–tháng OOS`;
    const gRows = [
      ["Nhóm YÊU (ASSD ≤ 5,9%)", g.loved, "badge-bull"],
      ["Nhóm TRUNG TÍNH", g.neutral, "badge-neutral"],
      ["Nhóm BỊ CHI PHỐI (Không yêu)", g.dominated, "badge-bear"],
      ["Cổ phiếu Vé số (Top 25% MAX)", g.lottery_high_max, "badge-neutral"],
    ];
    qs("#h3-groups-body").innerHTML = gRows.map(([lbl, obj, cls]) => `
      <tr>
        <td><span class="badge ${cls}">${lbl}</span></td>
        <td class="num">${obj.count}</td>
        <td class="num">${fmtPct(obj.fwd_ret_pct, 2)}</td>
        <td class="num">${obj.curr_cmf !== undefined ? fmtNum(obj.curr_cmf, 4) : `AbTurn ${fmtNum(obj.curr_abturn, 2)}x`}</td>
        <td class="num ${obj.fwd_cmf > -0.04 ? "pos" : "neg"}">${fmtNum(obj.fwd_cmf, 4)}</td>
        <td class="num">${fmtNum(obj.fwd_abturn, 2)}x</td>
      </tr>
    `).join("");

    qs("#h3-reg-body").innerHTML = BUNDLE.flow_analysis_h3.reg_fwd_cmf.map((r) => {
      const sig = r.p_value < 0.01 ? "Có ý nghĩa 1% (***)" : r.p_value < 0.05 ? "Có ý nghĩa 5% (**)" : r.p_value < 0.10 ? "Có ý nghĩa 10% (*)" : "Không có ý nghĩa";
      const sigCls = r.p_value < 0.05 ? "badge-bull" : "badge-neutral";
      return `<tr>
        <td><code>${r.variable}</code></td>
        <td class="num ${r.coef >= 0 ? "pos" : "neg"}">${fmtNum(r.coef, 4)}</td>
        <td class="num">${fmtNum(r.se, 4)}</td>
        <td class="num">${fmtNum(r.t_stat, 2)}</td>
        <td class="num">${fmtNum(r.p_value, 4)}</td>
        <td><span class="badge ${sigCls}">${sig}</span></td>
      </tr>`;
    }).join("");

    // Tab 4 Backtest
    const bt = BUNDLE.backtest_h4;
    qs("#bt-cost-sub").textContent = bt.cost_note;

    const seriesCfg = [
      { key: "ASSD-Opt (TS-QML)", short: "ASSD TS-QML", token: "bull", width: 2.8, dash: null, visible: true },
      { key: "Min-Volatility", short: "Min-Vol", token: "teal", width: 1.9, dash: "4 3", visible: true },
      { key: "HOSE Equal-Weight Basket", short: "HOSE Basket", token: "bench", width: 2.1, dash: "6 3", visible: true },
      { key: "Point-ML (Ridge Top-8)", short: "Ridge Top-8", token: "amber", width: 1.8, dash: null, visible: true },
      { key: "Mean-Variance (Markowitz)", short: "Markowitz", token: "violet", width: 1.7, dash: "2 3", visible: false },
      { key: "ASSD-Opt (QRF)", short: "ASSD QRF", token: "bear", width: 1.7, dash: null, visible: false },
      { key: "VN30 ETF (E1VFVN30)", short: "VN30 ETF", token: "muted", width: 1.7, dash: "5 2", visible: true },
    ];

    const renderBtCharts = () => {
      window.ChartEngine.renderMultiEquityAndDrawdown(qs("#svg-bt-eq"), qs("#tip-bt-eq"), qs("#svg-bt-dd"), bt, seriesCfg);
    };

    const legBox = qs("#bt-legend-toggles");
    legBox.innerHTML = "";
    seriesCfg.forEach((sc) => {
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = `btn btn-sm btn-toggle${sc.visible ? " is-active" : ""}`;
      btn.textContent = sc.key;
      btn.addEventListener("click", () => {
        sc.visible = !sc.visible;
        btn.classList.toggle("is-active", sc.visible);
        renderBtCharts();
      });
      legBox.appendChild(btn);
    });
    renderBtCharts();

    const btBody = qs("#bt-table-body");
    btBody.innerHTML = Object.entries(bt.metrics).map(([name, m]) => `
      <tr>
        <td><strong>${name}</strong></td>
        <td class="num ${m.cum_return_pct >= 0 ? "pos" : "neg"}">${fmtPct(m.cum_return_pct, 2)}</td>
        <td class="num">${fmtPct(m.ann_return_pct, 2)}</td>
        <td class="num">${fmtNum(m.sharpe)}</td>
        <td class="num">${fmtNum(m.sortino)}</td>
        <td class="num">${fmtNum(m.omega)}</td>
        <td class="num neg">${fmtPct(m.max_drawdown_pct, 2)}</td>
        <td class="num">${fmtPct(m.cvar5_pct, 2)}</td>
        <td class="num">${fmtPct(m.avg_turnover_pct, 1)}</td>
      </tr>
    `).join("");

    // Optimal Weights Allocation Grid
    const topAlloc = [...BUNDLE.stocks].filter((x) => x.opt_weight_pct >= 0.5).sort((a, b) => b.opt_weight_pct - a.opt_weight_pct);
    qs("#opt-alloc-grid").innerHTML = topAlloc.map((s) => `
      <div class="alloc-item">
        <div style="display:flex;justify-content:space-between;font-weight:700;">
          <span>${s.short_ticker}</span>
          <span class="pos">${fmtPct(s.opt_weight_pct, 1)}</span>
        </div>
        <div class="panel-sub">${s.sector}</div>
        <div class="bar-bg" style="margin-top:5px;"><div class="bar-fill" style="width:${Math.min(100, s.opt_weight_pct * 5)}%"></div></div>
      </div>
    `).join("");

    // Data Audit Log
    qs("#audit-badge-sub").textContent = `${BUNDLE.data_audit.total_repairs} phiên ngoài biên độ HOSE ±7% đã hiệu chỉnh`;
    qs("#audit-log-body").innerHTML = BUNDLE.data_audit.sample_repairs.map((ev) => `
      <tr>
        <td class="nowrap"><strong>${ev.ticker}</strong></td>
        <td class="nowrap">${ev.date}</td>
        <td class="num nowrap neg">${fmtPct(ev.raw_ret_pct, 2)}</td>
        <td class="num nowrap">${fmtPct(ev.repaired_ret_pct, 2)}</td>
        <td>${ev.action}</td>
      </tr>
    `).join("");
  }

  function renderSectorFlows() {
    const secMap = {};
    for (const s of BUNDLE.stocks) {
      const sv = getStockView(s);
      const cls = window.SDMath.classifySd(sv.sd, currentEps);
      if (!secMap[s.sector]) secMap[s.sector] = [];
      secMap[s.sector].push({ s, sv, cls });
    }
    const list = Object.entries(secMap).map(([sector, items]) => {
      const loved = items.filter((x) => x.cls.isLoved).length;
      const ratio = (100 * loved) / items.length;
      const avgEps = items.reduce((a, b) => a + b.sv.sd.eps2_star * 100, 0) / items.length;
      const avgAb = items.reduce((a, b) => a + b.s.ab_turn, 0) / items.length;
      const avgCmf = items.reduce((a, b) => a + b.s.cmf_20, 0) / items.length;
      const sumW = items.reduce((a, b) => a + b.s.opt_weight_pct, 0);
      return { sector, count: items.length, loved, ratio, avgEps, avgAb, avgCmf, sumW };
    }).sort((a, b) => b.ratio - a.ratio || a.avgEps - b.avgEps);

    qs("#sector-flow-body").innerHTML = list.map((sec) => {
      let badge = `<span class="badge badge-neutral">Trung tính</span>`;
      if (sec.ratio >= 50) badge = `<span class="badge badge-bull">Ưu thế &amp; Hút vốn</span>`;
      else if (sec.ratio < 20) badge = `<span class="badge badge-bear">Suy yếu / Bị chi phối</span>`;
      return `<tr>
        <td><strong>${sec.sector}</strong> (${sec.count} mã)</td>
        <td class="num">${sec.loved}/${sec.count} (${fmtPct(sec.ratio, 0)})<div class="bar-bg" style="margin-top:4px;"><div class="bar-fill" style="width:${sec.ratio}%"></div></div></td>
        <td class="num">${fmtPct(sec.avgEps, 2)}</td>
        <td class="num">${fmtNum(sec.avgAb, 2)}x</td>
        <td class="num ${sec.avgCmf >= 0 ? "pos" : "neg"}">${fmtNum(sec.avgCmf, 3)}</td>
        <td class="num">${fmtPct(sec.sumW, 1)}</td>
        <td>${badge}</td>
      </tr>`;
    }).join("");
  }

  function renderResearchTab() {
    const mQrf = BUNDLE.distributional_eval_h1["Quantile Regression Forests (Meinshausen 2006)"];
    const mHist = BUNDLE.distributional_eval_h1["Historical Empirical CDF Baseline"];
    const mAssd = BUNDLE.backtest_h4.metrics["ASSD-Opt (TS-QML)"];
    const mBench = BUNDLE.backtest_h4.metrics["HOSE Equal-Weight Basket"];
    const regCmf = BUNDLE.flow_analysis_h3.reg_fwd_cmf.find((r) => r.variable === "ASSD_Score_t");

    const hyps = [
      {
        code: "Giả thuyết H1 — Dự báo Phân phối (Distributional Advantage)",
        status: "ĐẠT THỰC NGHIỆM",
        cls: "badge-bull",
        body: `QRF đạt Pinball Loss ${fmtPct(mQrf.pinball_mean, 4)} & CRPS ${fmtPct(mQrf.crps_pct, 4)}, thấp hơn Historical Empirical CDF (${fmtPct(mHist.pinball_mean, 4)}) và độ phủ 90% đạt ${fmtPct(mQrf.coverage_90_pct, 1)}.`,
      },
      {
        code: "Giả thuyết H2 — Vai trò Almost SD vs FSD/SSD",
        status: "ĐẠT THỰC NGHIỆM",
        cls: "badge-bull",
        body: `SSD nghiêm ngặt bỏ sót nhiều mã tiềm năng do nhiễu đuôi; ASSD ở ngưỡng Levy 5,9% phân tách rõ 641 mẫu "Yêu", 464 "Trung tính" và 605 "Bị chi phối" trên 1.710 quan sát.`,
      },
      {
        code: "Giả thuyết H3 — Dòng tiền & Hiệu ứng Cổ phiếu Vé số",
        status: "ĐẠT Ý NGHĨA 5%",
        cls: "badge-bull",
        body: `Hồi quy White HC1 trên 1.710 mẫu cho thấy ASSD_Score_t dự báo CMF(t+1) với β = +${fmtNum(regCmf.coef, 4)} (t = ${fmtNum(regCmf.t_stat)}, p = ${fmtNum(regCmf.p_value, 4)}); cổ phiếu MAX vé số (AbTurn 1,33x) bị rút tiền kỳ sau (p < 0,001).`,
      },
      {
        code: "Giả thuyết H4 — Hiệu quả Danh mục Out-of-Sample",
        status: "VƯỢT TRỘI LỢI SUẤT & RỦI RO",
        cls: "badge-bench",
        body: `ASSD-Opt (TS-QML) đạt +${fmtPct(mAssd.cum_return_pct, 2)} (Sharpe ${fmtNum(mAssd.sharpe)}, Sortino ${fmtNum(mAssd.sortino)}, MaxDD ${fmtPct(mAssd.max_drawdown_pct, 2)}) vượt Basket +${fmtPct(mBench.cum_return_pct, 2)}; p-value bootstrap trên 45 tháng là ${fmtNum(mAssd.boot_p_value, 3)}.`,
      },
    ];

    qs("#hypotheses-grid").innerHTML = hyps.map((h) => `
      <div class="hyp-box">
        <div style="display:flex;justify-content:space-between;align-items:center;gap:6px;margin-bottom:6px;">
          <strong>${h.code}</strong>
          <span class="badge ${h.cls}">${h.status}</span>
        </div>
        <div class="panel-sub">${h.body}</div>
      </div>
    `).join("");

    renderPapers();
  }

  function renderPapers() {
    if (!CORPUS) return;
    const kw = qs("#paper-search").value.trim().toLowerCase();
    const br = qs("#paper-branch").value;
    const sortBy = qs("#paper-sort").value;

    let items = CORPUS.filter((p) => {
      if (br !== "ALL" && p.branch !== br) return false;
      if (kw) {
        const hay = `${p.title} ${p.authors} ${p.journal} ${p.doi}`.toLowerCase();
        if (!hay.includes(kw)) return false;
      }
      return true;
    });

    items = [...items].sort((a, b) => sortBy === "year" ? (b.year - a.year || b.citations - a.citations) : (b.citations - a.citations || b.year - a.year));
    qs("#corpus-count-label").textContent = `${items.length} / ${CORPUS.length} bài Q1`;

    qs("#paper-list").innerHTML = items.slice(0, 60).map((p) => `
      <div class="paper-row">
        <div class="paper-title">${p.title}</div>
        <div class="paper-meta">
          <strong>${p.authors}</strong> (${p.year}) · <em>${p.journal}</em> · Trích dẫn: <strong>${p.citations}</strong> · Nhánh: <code>${p.branch}</code> · <a href="${p.url}" target="_blank" rel="noopener" style="color:var(--bench);">DOI: ${p.doi}</a>
        </div>
        ${p.abstract ? `<div class="paper-abs">${p.abstract}...</div>` : ""}
      </div>
    `).join("");
  }

  function runBrowserMlTab() {
    if (!PANEL) return;
    const btn = qs("#btn-run-liveml");
    btn.disabled = true;
    try {
      const res = window.BrowserML.runLiveExperiment(PANEL, {
        algo: qs("#live-algo").value,
        model: qs("#live-algo").value,
        horizon: Number(qs("#live-horizon").value),
        wMax: Number(qs("#live-wmax").value),
        txCostPct: Number(qs("#live-tx").value),
        lotteryFilter: qs("#live-lottery").checked,
        eps: currentEps,
      });

      qs("#liveml-runtime").textContent = `Đã huấn luyện xong trong ${res.elapsedMs} ms (${res.nRebalances} kỳ Walk-Forward × 38 mã)`;
      qs("#liveml-status-text").textContent = `ASSD-Opt: ${fmtPct(res.stats.assd.cum_return_pct, 1)} (Sharpe ${fmtNum(res.stats.assd.sharpe)}) | Pinball ML: ${fmtPct(res.pinballMeanPct, 3)} vs Hist: ${fmtPct(res.pinballHistPct, 3)}`;

      window.ChartEngine.renderSimpleMultiChart(
        qs("#svg-live-eq"), qs("#tip-live-eq"), res.dates,
        [
          { label: "ASSD Live Browser", vals: res.curves.assd, token: "bull", width: 2.6 },
          { label: "Min-Vol Phòng thủ", vals: res.curves.invVol, token: "teal", width: 1.8, dash: "4 3" },
          { label: "HOSE Basket", vals: res.curves.bench, token: "bench", width: 2.0, dash: "6 3" },
        ],
        `Live Browser Walk-Forward (T+${res.horizon}, 520 phiên gần nhất)`,
        "Giá trị (tr. VND)", false
      );

      window.ChartEngine.renderSimpleMultiChart(
        qs("#svg-live-pinball"), qs("#tip-live-pinball"),
        res.pinballByTau.map((x) => x.tau),
        [
          { label: "Quantile ML (Live)", vals: res.pinballByTau.map((x) => x.model), token: "bull" },
          { label: "Historical Empirical", vals: res.pinballByTau.map((x) => x.hist), token: "bench" },
        ],
        "Pinball Loss theo từng Phân vị τ (Thấp hơn là tốt hơn)",
        "Pinball (%)", true
      );

      qs("#liveml-stock-body").innerHTML = res.stocks.map((st) => `
        <tr>
          <td><strong>${st.ticker}</strong></td>
          <td>${st.sector}</td>
          <td class="num">${fmtPct(st.q10_pct, 1)}</td>
          <td class="num">${fmtPct(st.q50_pct, 1)}</td>
          <td class="num">${fmtPct(st.q90_pct, 1)}</td>
          <td class="num">${fmtPct(st.eps2_star * 100, 2)}</td>
          <td class="num pos"><strong>${fmtPct(st.weight_pct, 1)}</strong></td>
          <td><span class="badge ${st.verdict_cls}">${st.verdict}</span></td>
        </tr>
      `).join("");
      liveMlRanOnce = true;
    } finally {
      btn.disabled = false;
    }
  }

  function activateTab(targetId) {
    qsa(".nav-tab").forEach((b) => {
      const act = b.dataset.tab === targetId;
      b.classList.toggle("is-active", act);
      b.setAttribute("aria-selected", act ? "true" : "false");
    });
    qsa(".tab-panel").forEach((p) => {
      p.classList.toggle("is-hidden", p.id !== `tab-${targetId}`);
    });
    if (targetId === "liveml" && !liveMlRanOnce && PANEL) {
      runBrowserMlTab();
    }
  }

  function wireEvents() {
    qs("#btn-theme-toggle").addEventListener("click", () => {
      const cur = document.documentElement.getAttribute("data-theme");
      const next = cur === "light" ? "dark" : "light";
      if (next === "light") document.documentElement.setAttribute("data-theme", "light");
      else document.documentElement.removeAttribute("data-theme");
      qs("#theme-lbl").textContent = next === "light" ? "Nền Tối" : "Nền Sáng";
      if (BUNDLE) {
        renderDetailTerminal();
        renderH1AndFlowsAndBacktest();
        if (liveMlRanOnce) runBrowserMlTab();
      }
    });

    qs("#model-select").addEventListener("change", (e) => {
      activeModel = e.target.value;
      renderTickerRibbon();
      renderKPIs();
      renderScreener();
      renderDetailTerminal();
      renderSectorFlows();
    });

    const updateEps = (val) => {
      currentEps = Number(val);
      qs("#eps-slider").value = currentEps.toFixed(3);
      qs("#eps-readout").textContent = `${currentEps.toFixed(3)} (${fmtPct(currentEps * 100, 1)})`;
      qsa("[data-eps]").forEach((b) => b.classList.toggle("is-active", Math.abs(Number(b.dataset.eps) - currentEps) < 1e-4));
      renderTickerRibbon();
      renderKPIs();
      renderScreener();
      renderDetailTerminal();
      renderSectorFlows();
    };

    qs("#eps-slider").addEventListener("input", (e) => updateEps(e.target.value));
    qsa("[data-eps]").forEach((b) => b.addEventListener("click", () => updateEps(b.dataset.eps)));

    qsa(".nav-tab").forEach((b) => b.addEventListener("click", () => activateTab(b.dataset.tab)));

    qs("#search-ticker").addEventListener("input", renderScreener);
    qs("#sector-filter").addEventListener("change", renderScreener);
    qs("#verdict-filter").addEventListener("change", renderScreener);
    qs("#compare-target").addEventListener("change", (e) => {
      compareTarget = e.target.value;
      renderDetailTerminal();
    });

    qsa("th.sortable").forEach((th) => {
      th.addEventListener("click", () => {
        const col = th.dataset.sort;
        if (sortCol === col) sortAsc = !sortAsc;
        else {
          sortCol = col;
          sortAsc = col === "eps2" || col === "ticker";
        }
        renderScreener();
      });
    });

    qs("#btn-run-liveml").addEventListener("click", runBrowserMlTab);

    qs("#paper-search").addEventListener("input", renderPapers);
    qs("#paper-branch").addEventListener("change", renderPapers);
    qs("#paper-sort").addEventListener("change", renderPapers);

    qs("#btn-export-csv").addEventListener("click", () => {
      if (!BUNDLE) return;
      const lines = ["Ticker,Name,Sector,LastClose,PredMeanPct,Eps2Pct,CMF20,AbTurn,ASSDWeightPct,Verdict"];
      BUNDLE.stocks.forEach((s) => {
        const sv = getStockView(s);
        const cls = window.SDMath.classifySd(sv.sd, currentEps);
        lines.push(`${s.short_ticker},"${s.name}","${s.sector}",${s.last_close},${sv.meanPct.toFixed(2)},${(sv.sd.eps2_star * 100).toFixed(2)},${s.cmf_20},${s.ab_turn},${s.opt_weight_pct},"${cls.label}"`);
      });
      const blob = new Blob(["\uFEFF" + lines.join("\n")], { type: "text/csv;charset=utf-8;" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = `VN_HOSE_ASSD_ML_Ranking_${activeModel}.csv`;
      a.click();
    });

    // Sample CSV generator + Custom CSV Analyzer
    qs("#btn-sample-csv").addEventListener("click", () => {
      if (!PANEL) return;
      const idx = 6; // FPT
      const rows = ["Date,Close,High,Low,Volume"];
      for (let i = 0; i < PANEL.dates.length; i++) {
        const c = PANEL.closes[i][idx];
        rows.push(`${PANEL.dates[i]},${c},${(c * 1.015).toFixed(1)},${(c * 0.985).toFixed(1)},${Math.round(2200000 * PANEL.abturn[i][idx])}`);
      }
      const blob = new Blob([rows.join("\n")], { type: "text/csv;charset=utf-8;" });
      const a = document.createElement("a");
      a.href = URL.createObjectURL(blob);
      a.download = "mau_FPT_HOSE_520d.csv";
      a.click();
    });

    qs("#csv-upload").addEventListener("change", async (e) => {
      const file = e.target.files[0];
      if (!file || !BUNDLE) return;
      const box = qs("#custom-csv-result");
      try {
        const text = await file.text();
        const res = window.BrowserML.analyzeCustomCsv(text, BUNDLE.benchmark.cdf_ts, BUNDLE.grid_pct, currentEps);
        box.style.display = "block";
        box.innerHTML = `
          <div><strong>Kết quả Phân tích ML + ASSD cho file <code>${file.name}</code>:</strong> <span class="badge ${res.verdict.cls}">${res.verdict.label}</span></div>
          <div class="formula-mono">
            • Dữ liệu: ${res.nDays} phiên (${res.start} &rarr; ${res.end}) · Đã hiệu chỉnh ${res.repairs} phiên nhảy giá chia tách<br/>
            • Dự báo ML 22 phiên tới: Q10 = ${fmtPct(res.q10Pct, 2)} | Q50 = ${fmtPct(res.q50Pct, 2)} | Q90 = ${fmtPct(res.q90Pct, 2)} | E[R] = <strong>${fmtPct(res.predMeanPct, 2)}</strong><br/>
            • Kiểm định ASSD vs VN-Basket: <strong>&epsilon;<sub>2</sub>* = ${fmtPct(res.sd.eps2_star * 100, 2)}</strong> · Dòng tiền CMF-20: ${fmtNum(res.cmf20, 3)}${res.avgFnf !== null ? ` · Khối ngoại TB: ${fmtNum(res.avgFnf, 3)}` : ""}
          </div>
        `;
      } catch (err) {
        box.style.display = "block";
        box.innerHTML = `<span class="neg">Lỗi đọc CSV: ${err.message}</span>`;
      }
    });
  }

  async function init() {
    wireEvents();
    try {
      const [rBundle, rPanel, rCorpus] = await Promise.all([
        fetch("./data/research_bundle.json"),
        fetch("./data/market_panel.json"),
        fetch("./data/q1_corpus.json"),
      ]);
      if (!rBundle.ok) throw new Error(`Không tải được research_bundle.json (${rBundle.status})`);
      BUNDLE = await rBundle.json();
      PANEL = rPanel.ok ? await rPanel.json() : null;
      CORPUS = rCorpus.ok ? await rCorpus.json() : [];

      qs("#hdr-date-chip").textContent = `${BUNDLE.date_range.start} → ${BUNDLE.date_range.end} (${BUNDLE.date_range.trading_days} phiên · ${BUNDLE.universe_count} mã)`;
      populateDropdowns();
      renderTickerRibbon();
      renderKPIs();
      renderScreener();
      renderDetailTerminal();
      renderH1AndFlowsAndBacktest();
      renderResearchTab();
      hideStatus();
    } catch (err) {
      showStatus(`Lỗi khởi tạo dữ liệu: ${err.message}`, true);
    }
  }

  init();
})();
