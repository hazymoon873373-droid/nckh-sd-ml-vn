(() => {
  "use strict";

  let MODEL = null;
  let selectedTicker = "FPT.VN";
  let currentEps = 0.059;

  const qs = (sel) => document.querySelector(sel);
  const qsa = (sel) => Array.from(document.querySelectorAll(sel));

  function fmtPct(num, digits = 2) {
    if (num === null || num === undefined || Number.isNaN(num)) return "--";
    return num.toFixed(digits).replace(".", ",") + "%";
  }

  function fmtNum(num, digits = 2) {
    if (num === null || num === undefined || Number.isNaN(num)) return "--";
    return num.toFixed(digits).replace(".", ",");
  }

  function classifyStock(stock, eps) {
    const sd = stock.sd;
    if (sd.ssd_strict) return { text: "YÊU (SSD Chuẩn)", cls: "badge-love", isLoved: true };
    if (sd.mean_ok && sd.eps2_star <= eps) return { text: "YÊU (Almost SD)", cls: "badge-love", isLoved: true };
    if (!sd.mean_ok && sd.eps2_reverse <= eps) return { text: "KHÔNG YÊU (Bị chi phối)", cls: "badge-hate", isLoved: false };
    return { text: "TRUNG TÍNH", cls: "badge-neutral", isLoved: false };
  }

  function renderKPIs() {
    if (!MODEL) return;
    const stocks = MODEL.stocks;
    const classified = stocks.map((s) => ({ ...s, pref: classifyStock(s, currentEps) }));
    const loved = classified.filter((s) => s.pref.isLoved);

    qs("#kpi-loved").textContent = `${loved.length} / ${stocks.length}`;
    qs("#kpi-loved-sub").textContent = `Chiếm ${fmtPct((100 * loved.length) / stocks.length, 1)} rổ mã ở ngưỡng ε₂ = ${fmtPct(currentEps * 100, 1)}`;

    // Recompute sector leadership dynamically for currentEps
    const secMap = {};
    for (const s of classified) {
      if (!secMap[s.sector]) secMap[s.sector] = [];
      secMap[s.sector].push(s);
    }
    const secList = Object.entries(secMap).map(([sec, items]) => {
      const nLoved = items.filter((x) => x.pref.isLoved).length;
      const ratio = (100 * nLoved) / items.length;
      const avgEps = items.reduce((acc, x) => acc + x.sd.eps2_star, 0) / items.length;
      return { sec, nLoved, total: items.length, ratio, avgEps };
    }).sort((a, b) => b.ratio - a.ratio || a.avgEps - b.avgEps);

    if (secList.length) {
      const top = secList[0];
      qs("#kpi-top-sector").textContent = top.sec;
      qs("#kpi-top-sector-sub").textContent = `${top.nLoved}/${top.total} mã "Yêu" (${fmtPct(top.ratio, 0)}) · ε₂* TB ${fmtPct(top.avgEps * 100, 2)}`;
    }

    const mAssd = MODEL.backtest.metrics["ASSD + QRF (Top-5 Yêu)"];
    const mBench = MODEL.backtest.metrics["VN-Basket Benchmark"];
    if (mAssd && mBench) {
      qs("#kpi-bt-ret").textContent = fmtPct(mAssd.cum_return_pct, 1);
      qs("#kpi-bt-sub").textContent = `VN-Basket: ${fmtPct(mBench.cum_return_pct, 1)} (${MODEL.backtest.n_rebalances} kỳ OOS)`;
      qs("#kpi-bt-mdd").textContent = fmtPct(mAssd.max_drawdown_pct, 1);
      qs("#kpi-bt-sortino").textContent = `Sortino ${fmtNum(mAssd.sortino)} · Omega ${fmtNum(mAssd.omega)} (Bench Sortino ${fmtNum(mBench.sortino)})`;
    }
  }

  function renderStockTable() {
    const tbody = qs("#stock-table-body");
    tbody.innerHTML = "";
    for (const s of MODEL.stocks) {
      const pref = classifyStock(s, currentEps);
      const tr = document.createElement("tr");
      tr.className = "stock-row" + (s.ticker === selectedTicker ? " is-selected" : "");
      tr.tabIndex = 0;
      tr.setAttribute("role", "button");
      tr.setAttribute("aria-label", `Chọn cổ phiếu ${s.short_ticker}`);
      tr.innerHTML = `
        <td><strong>${s.short_ticker}</strong><br><span class="panel-note">${s.name}</span></td>
        <td>${s.sector}</td>
        <td class="num">${fmtPct(s.pred_mean_pct, 2)}</td>
        <td class="num">${fmtPct(s.sd.eps2_star * 100, 2)}</td>
        <td class="num">${fmtNum(s.cmf_20, 2)}</td>
        <td><span class="badge ${pref.cls}">${pref.text}</span></td>
      `;
      tr.addEventListener("click", () => selectStock(s.ticker));
      tr.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          selectStock(s.ticker);
        }
      });
      tbody.appendChild(tr);
    }
  }

  function drawLineChart(svgEl, opts) {
    const {
      xVals, ySeries, xLabel, yLabel, title,
      yMin, yMax, shadeBetween = null, zeroLine = false, isPctX = true,
      height = 240, leftPad = null
    } = opts;
    const W = 640, H = height;
    const pad = { l: leftPad || (isPctX ? 58 : 74), r: 20, t: 28, b: 38 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const xMin = xVals[0], xMax = xVals[xVals.length - 1];
    const sx = (v) => pad.l + ((v - xMin) / (xMax - xMin || 1)) * pw;
    const sy = (v) => pad.t + ph - ((v - yMin) / (yMax - yMin || 1)) * ph;

    let parts = [];
    // Grid lines
    for (let k = 0; k <= 4; k++) {
      const yv = yMin + (k / 4) * (yMax - yMin);
      const py = sy(yv);
      parts.push(`<line x1="${pad.l}" y1="${py.toFixed(1)}" x2="${W - pad.r}" y2="${py.toFixed(1)}" stroke="#e3ded3" stroke-width="1" />`);
      parts.push(`<text x="${pad.l - 8}" y="${(py + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="#525d70" font-family="var(--font-mono)">${yv.toFixed(2)}</text>`);
    }

    const xTicks = 5;
    for (let k = 0; k <= xTicks; k++) {
      const xv = xMin + (k / xTicks) * (xMax - xMin);
      const px = sx(xv);
      parts.push(`<line x1="${px.toFixed(1)}" y1="${pad.t}" x2="${px.toFixed(1)}" y2="${H - pad.b}" stroke="#eeeae0" stroke-width="1" />`);
      const lbl = isPctX ? `${xv.toFixed(0)}%` : `${xv}`;
      parts.push(`<text x="${px.toFixed(1)}" y="${H - pad.b + 16}" text-anchor="middle" font-size="11" fill="#525d70" font-family="var(--font-mono)">${lbl}</text>`);
    }

    if (zeroLine && xMin < 0 && xMax > 0) {
      const xz = sx(0);
      parts.push(`<line x1="${xz.toFixed(1)}" y1="${pad.t}" x2="${xz.toFixed(1)}" y2="${H - pad.b}" stroke="#9ca3af" stroke-dasharray="4 3" stroke-width="1" />`);
    }

    // Optional shaded violation area where series[0] > series[1] (S2 region)
    if (shadeBetween) {
      const [aVals, bVals, fillCol] = shadeBetween;
      let pathPts = [];
      let inRegion = false;
      for (let i = 0; i < xVals.length; i++) {
        if (aVals[i] > bVals[i]) {
          if (!inRegion) {
            inRegion = true;
            pathPts = [{ x: sx(xVals[i]), yTop: sy(aVals[i]), yBot: sy(bVals[i]) }];
          } else {
            pathPts.push({ x: sx(xVals[i]), yTop: sy(aVals[i]), yBot: sy(bVals[i]) });
          }
        } else if (inRegion) {
          const topStr = pathPts.map((p, idx) => `${idx === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.yTop.toFixed(1)}`).join(" ");
          const botStr = pathPts.slice().reverse().map((p) => `L${p.x.toFixed(1)},${p.yBot.toFixed(1)}`).join(" ");
          parts.push(`<path d="${topStr} ${botStr} Z" fill="${fillCol}" opacity="0.45" />`);
          inRegion = false;
        }
      }
      if (inRegion && pathPts.length > 1) {
        const topStr = pathPts.map((p, idx) => `${idx === 0 ? "M" : "L"}${p.x.toFixed(1)},${p.yTop.toFixed(1)}`).join(" ");
        const botStr = pathPts.slice().reverse().map((p) => `L${p.x.toFixed(1)},${p.yBot.toFixed(1)}`).join(" ");
        parts.push(`<path d="${topStr} ${botStr} Z" fill="${fillCol}" opacity="0.45" />`);
      }
    }

    // Lines
    for (const ser of ySeries) {
      const d = xVals.map((xv, i) => `${i === 0 ? "M" : "L"}${sx(xv).toFixed(1)},${sy(ser.vals[i]).toFixed(1)}`).join(" ");
      parts.push(`<path d="${d}" fill="none" stroke="${ser.color}" stroke-width="${ser.width || 2.2}" ${ser.dash ? `stroke-dasharray="${ser.dash}"` : ""} />`);
    }

    // Titles & axes labels
    parts.push(`<text x="${pad.l}" y="17" font-size="12" font-weight="700" fill="#151922">${title}</text>`);
    parts.push(`<text x="${(pad.l + (W - pad.r)) / 2}" y="${H - 6}" text-anchor="middle" font-size="11" fill="#525d70">${xLabel}</text>`);
    parts.push(`<text x="13" y="${(pad.t + (H - pad.b)) / 2}" text-anchor="middle" font-size="11" fill="#525d70" transform="rotate(-90 13 ${(pad.t + (H - pad.b)) / 2})">${yLabel}</text>`);

    svgEl.innerHTML = parts.join("");
  }

  function renderSelectedStockCharts() {
    const stock = MODEL.stocks.find((s) => s.ticker === selectedTicker) || MODEL.stocks[0];
    selectedTicker = stock.ticker;
    const pref = classifyStock(stock, currentEps);

    qs("#chart-title").textContent = `Phân tích CDF & Almost SD: ${stock.short_ticker} (${stock.name}) vs VN-Leaders`;
    const badge = qs("#selected-badge");
    badge.className = `badge ${pref.cls}`;
    badge.textContent = pref.text;

    const grid = MODEL.grid_pct;
    drawLineChart(qs("#svg-cdf"), {
      xVals: grid,
      ySeries: [
        { vals: MODEL.benchmark.cdf, color: "#1d4ed8", width: 2.0, dash: "5 3" },
        { vals: stock.cdf, color: "#0d6e4f", width: 2.6 },
      ],
      xLabel: "Lợi suất dự báo 22 phiên tới r (%)",
      yLabel: "CDF F(r)",
      title: `1. Đường CDF Tương lai F(r): ${stock.short_ticker} nằm dưới/bên phải Benchmark là tốt hơn`,
      yMin: 0,
      yMax: 1.0,
      zeroLine: true,
      shadeBetween: [stock.cdf, MODEL.benchmark.cdf, "#f5b7b1"],
    });

    const f2Max = Math.max(...stock.sd.f2_a, ...stock.sd.f2_b, 0.01);
    drawLineChart(qs("#svg-f2"), {
      xVals: grid,
      ySeries: [
        { vals: stock.sd.f2_b, color: "#1d4ed8", width: 2.0, dash: "5 3" },
        { vals: stock.sd.f2_a, color: "#0d6e4f", width: 2.6 },
      ],
      xLabel: "Ngưỡng lợi suất t (%)",
      yLabel: "Tích phân bậc 2 F(2)(t)",
      title: `2. Kiểm định SSD / Almost SD trên F(2)(t): Vùng tô đỏ là diện tích vi phạm S2`,
      yMin: 0,
      yMax: f2Max * 1.08,
      zeroLine: true,
      shadeBetween: [stock.sd.f2_a, stock.sd.f2_b, "#f5b7b1"],
    });

    const sd = stock.sd;
    const box = qs("#live-formula-box");
    box.innerHTML = `
      <div><strong>Tính toán ASSD trực tiếp cho ${stock.short_ticker} (Tzeng, Huang &amp; Shih, 2013):</strong></div>
      <div class="formula-math">
        1) Lợi suất kỳ vọng QRF: E[R_${stock.short_ticker}] = ${fmtPct(stock.pred_mean_pct, 2)} vs E[R_Bench] = ${fmtPct(stock.bench_pred_mean_pct, 2)} &rarr; <strong>${sd.mean_ok ? "ĐẠT (lớn hơn chuẩn)" : "KHÔNG ĐẠT"}</strong><br/>
        2) Tỷ lệ diện tích vi phạm ASSD: &epsilon;&#8322;* = ${sd.area_viol2.toFixed(6)} / ${sd.area_total2.toFixed(6)} = <strong>${fmtPct(sd.eps2_star * 100, 2)}</strong> (Ngưỡng đang chọn: ${fmtPct(currentEps * 100, 1)})<br/>
        3) Tỷ lệ vi phạm FSD (&epsilon;&#8321;*): ${fmtPct(sd.eps1_star * 100, 2)} · Xác suất lỗ 22 phiên: ${fmtPct(stock.loss_prob_pct, 1)} · Thanh khoản AbTurn: ${fmtNum(stock.ab_turn, 2)}x
      </div>
    `;
  }

  function renderFlowTab() {
    const secMap = {};
    for (const s of MODEL.stocks) {
      const pref = classifyStock(s, currentEps);
      if (!secMap[s.sector]) secMap[s.sector] = [];
      secMap[s.sector].push({ ...s, pref });
    }
    const sectors = Object.entries(secMap).map(([sector, items]) => {
      const nPref = items.filter((x) => x.pref.isLoved).length;
      const ratio = (100 * nPref) / items.length;
      const avgEps = (items.reduce((a, b) => a + b.sd.eps2_star, 0) / items.length) * 100;
      const avgAb = items.reduce((a, b) => a + b.ab_turn, 0) / items.length;
      const avgCmf = items.reduce((a, b) => a + b.cmf_20, 0) / items.length;
      return { sector, count: items.length, nPref, ratio, avgEps, avgAb, avgCmf, items };
    }).sort((a, b) => b.ratio - a.ratio || a.avgEps - b.avgEps);

    const tbody = qs("#sector-table-body");
    tbody.innerHTML = "";
    for (const sec of sectors) {
      let flowSig = "Trung tính / Theo dõi";
      let sigCls = "badge-neutral";
      if (sec.ratio >= 50 && sec.avgCmf > 0) {
        flowSig = "Dòng tiền & ASSD đồng pha TÍCH CỰC";
        sigCls = "badge-love";
      } else if (sec.ratio < 30 && sec.avgCmf < 0) {
        flowSig = "Rủi ro rút tiền / Bị chi phối";
        sigCls = "badge-hate";
      } else if (sec.avgAb >= 1.15 && sec.ratio < 40) {
        flowSig = "Tiền nóng vào nhưng ASSD chưa chuộng";
        sigCls = "badge-neutral";
      }
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${sec.sector}</strong><br><span class="panel-note">${sec.items.map((i) => i.short_ticker).join(", ")}</span></td>
        <td class="num">${sec.nPref}/${sec.count} (${fmtPct(sec.ratio, 0)})<div class="bar-track" style="margin-top:4px;"><div class="bar-fill-love" style="width:${sec.ratio}%"></div></div></td>
        <td class="num">${fmtPct(sec.avgEps, 2)}</td>
        <td class="num">${fmtNum(sec.avgAb, 2)}x</td>
        <td class="num">${fmtNum(sec.avgCmf, 3)}</td>
        <td><span class="badge ${sigCls}">${flowSig}</span></td>
      `;
      tbody.appendChild(tr);
    }

    const oos = MODEL.oos_flow_validation;
    qs("#oos-count-note").textContent = `${oos.n_stock_months} quan sát cổ phiếu–tháng OOS`;
    qs("#oos-flow-body").innerHTML = `
      <tr>
        <td>Lợi suất thực tế tháng kế tiếp (t+1)</td>
        <td class="num"><strong>${fmtPct(oos.preferred_avg_fwd_ret_pct, 2)}</strong></td>
        <td class="num">${fmtPct(oos.non_preferred_avg_fwd_ret_pct, 2)}</td>
      </tr>
      <tr>
        <td>Chỉ số dòng tiền Chaikin (CMF-20) kỳ t+1</td>
        <td class="num"><strong>${fmtNum(oos.preferred_avg_fwd_cmf, 3)}</strong></td>
        <td class="num">${fmtNum(oos.non_preferred_avg_fwd_cmf, 3)}</td>
      </tr>
      <tr>
        <td>Thanh khoản bất thường (AbTurn) kỳ t+1</td>
        <td class="num"><strong>${fmtNum(oos.preferred_avg_fwd_abturn, 2)}x</strong></td>
        <td class="num">${fmtNum(oos.non_preferred_avg_fwd_abturn, 2)}x</td>
      </tr>
    `;
  }

  function renderBacktestTab() {
    const bt = MODEL.backtest;
    qs("#bt-cost-note").textContent = bt.cost_note;

    const curveNames = [
      ["ASSD + QRF (Top-5 Yêu)", "#0d6e4f", 2.8, null],
      ["Mean-ML (Ridge Top-5)", "#d97706", 2.0, null],
      ["Min-Volatility (Top-5)", "#0f766e", 2.0, "4 3"],
      ["VN-Basket Benchmark", "#1d4ed8", 2.2, "6 3"],
    ];

    const allVals = Object.values(bt.curves).flat();
    const yMin = Math.min(...allVals) * 0.96;
    const yMax = Math.max(...allVals) * 1.04;
    const xIdx = bt.dates.map((_, i) => i);

    drawLineChart(qs("#svg-equity"), {
      xVals: xIdx,
      ySeries: curveNames.map(([name, color, width, dash]) => ({
        vals: bt.curves[name],
        color,
        width,
        dash,
      })),
      xLabel: `Kỳ tái cân bằng hàng tháng (OOS: ${bt.dates[1] || ""} đến ${bt.dates[bt.dates.length - 1] || ""})`,
      yLabel: "Giá trị danh mục (triệu VND)",
      title: "Hiệu quả tích lũy Out-of-Sample (Khởi đầu = 100 triệu VND)",
      yMin,
      yMax,
      isPctX: false,
      height: 270,
      leftPad: 74,
    });

    const mb = qs("#bt-metrics-body");
    mb.innerHTML = "";
    for (const [name] of curveNames) {
      const m = bt.metrics[name];
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td><strong>${name}</strong></td>
        <td class="num">${fmtPct(m.cum_return_pct, 2)}</td>
        <td class="num">${fmtNum(m.sharpe, 2)}</td>
        <td class="num">${fmtNum(m.sortino, 2)}</td>
        <td class="num">${fmtNum(m.omega, 2)}</td>
        <td class="num">${fmtPct(m.max_drawdown_pct, 2)}</td>
      `;
      mb.appendChild(tr);
    }

    const audit = MODEL.data_audit;
    qs("#audit-count-note").textContent = `Đã phát hiện & hiệu chỉnh ${audit.corporate_action_repairs_count} phiên lệch ngày chia tách/cổ tức`;
    const ab = qs("#audit-table-body");
    ab.innerHTML = "";
    for (const ev of audit.sample_repairs) {
      const tr = document.createElement("tr");
      tr.innerHTML = `
        <td class="nowrap"><strong>${ev.ticker}</strong></td>
        <td class="nowrap">${ev.date}</td>
        <td class="num nowrap">${fmtPct(ev.raw_daily_ret_pct, 2)}</td>
        <td>${ev.action}</td>
      `;
      ab.appendChild(tr);
    }
  }

  function renderAll() {
    qs("#data-range-pill").textContent = `Dữ liệu: ${MODEL.date_range.start} → ${MODEL.date_range.end} (${MODEL.date_range.trading_days} phiên)`;
    renderKPIs();
    renderStockTable();
    renderSelectedStockCharts();
    renderFlowTab();
    renderBacktestTab();
  }

  function selectStock(ticker) {
    selectedTicker = ticker;
    renderStockTable();
    renderSelectedStockCharts();
  }

  async function loadOverview() {
    try {
      const resp = await fetch("/api/overview");
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      MODEL = await resp.json();
      if (!MODEL.stocks.some((s) => s.ticker === selectedTicker) && MODEL.stocks.length) {
        selectedTicker = MODEL.stocks[0].ticker;
      }
      renderAll();
    } catch (err) {
      const box = qs("#error-box");
      box.classList.remove("panel-hidden");
      box.textContent = `Không tải được dữ liệu mô hình: ${err.message}`;
    }
  }

  function wireControls() {
    const slider = qs("#eps-slider");
    const readout = qs("#eps-readout");

    const updateEps = (val) => {
      currentEps = Number(val);
      slider.value = currentEps.toFixed(3);
      readout.textContent = `${currentEps.toFixed(3)} (${fmtPct(currentEps * 100, 1)})`;
      qsa(".preset-group .btn-ghost").forEach((b) => {
        b.classList.toggle("is-active", Math.abs(Number(b.dataset.eps) - currentEps) < 1e-4);
      });
      if (MODEL) {
        renderKPIs();
        renderStockTable();
        renderSelectedStockCharts();
        renderFlowTab();
      }
    };

    slider.addEventListener("input", (e) => updateEps(e.target.value));
    qsa(".preset-group .btn-ghost").forEach((btn) => {
      btn.addEventListener("click", () => updateEps(btn.dataset.eps));
    });

    qsa(".tab-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.dataset.tab;
        qsa(".tab-btn").forEach((b) => {
          const active = b === btn;
          b.classList.toggle("is-active", active);
          b.setAttribute("aria-selected", active ? "true" : "false");
        });
        ["lab", "flow", "backtest"].forEach((t) => {
          qs(`#tab-${t}`).classList.toggle("panel-hidden", t !== target);
        });
      });
    });

    qs("#refresh-btn").addEventListener("click", async () => {
      const btn = qs("#refresh-btn");
      btn.disabled = true;
      btn.textContent = "Đang tải Yahoo Finance & chạy lại QRF...";
      try {
        const resp = await fetch("/api/refresh", { method: "POST" });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        MODEL = await resp.json();
        renderAll();
      } catch (err) {
        const box = qs("#error-box");
        box.classList.remove("panel-hidden");
        box.textContent = `Lỗi khi cập nhật dữ liệu mới: ${err.message}`;
      } finally {
        btn.disabled = false;
        btn.textContent = "Cập nhật dữ liệu Yahoo (.VN)";
      }
    });
  }

  wireControls();
  loadOverview();
})();
