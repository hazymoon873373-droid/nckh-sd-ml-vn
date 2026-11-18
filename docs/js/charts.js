/**
 * charts.js — Institutional Financial SVG Charting Engine with Dynamic Theme Palette
 * and Mouse Hover Crosshair Tooltips. Zero literal hex colors — reads live CSS tokens.
 */
window.ChartEngine = (() => {
  "use strict";

  function pal() {
    const cs = getComputedStyle(document.documentElement);
    const g = (v, fb) => cs.getPropertyValue(v).trim() || fb;
    return {
      bg: g("--chart-bg", "#0d121d"),
      grid: g("--grid-line", "#1d2638"),
      ink: g("--ink", "#ecf1f9"),
      muted: g("--ink-secondary", "#98a6be"),
      bull: g("--bull", "#10b981"),
      bear: g("--bear", "#f43f5e"),
      amber: g("--neutral", "#f59e0b"),
      bench: g("--bench", "#3b82f6"),
      violet: g("--violet-accent", "#8b5cf6"),
      teal: g("--teal-accent", "#14b8a6"),
      border: g("--border-strong", "#344361"),
    };
  }

  function fmt(n, d = 2) {
    return Number(n).toFixed(d).replace(".", ",");
  }

  function attachCrosshair(svgEl, tooltipEl, xVals, pad, W, H, renderTipHtml) {
    const overlay = document.createElementNS("http://www.w3.org/2000/svg", "rect");
    overlay.setAttribute("x", pad.l);
    overlay.setAttribute("y", pad.t);
    overlay.setAttribute("width", Math.max(1, W - pad.l - pad.r));
    overlay.setAttribute("height", Math.max(1, H - pad.t - pad.b));
    overlay.setAttribute("fill", "transparent");
    overlay.style.cursor = "crosshair";

    const vline = document.createElementNS("http://www.w3.org/2000/svg", "line");
    vline.setAttribute("y1", pad.t);
    vline.setAttribute("y2", H - pad.b);
    vline.setAttribute("stroke", pal().border);
    vline.setAttribute("stroke-dasharray", "3 3");
    vline.setAttribute("stroke-width", "1");
    vline.style.display = "none";

    svgEl.appendChild(vline);
    svgEl.appendChild(overlay);

    overlay.addEventListener("mousemove", (evt) => {
      const rect = svgEl.getBoundingClientRect();
      const scaleX = W / rect.width;
      const mx = (evt.clientX - rect.left) * scaleX;
      const pw = W - pad.l - pad.r;
      const ratio = Math.max(0, Math.min(1, (mx - pad.l) / pw));
      const idx = Math.round(ratio * (xVals.length - 1));
      const px = pad.l + (idx / Math.max(1, xVals.length - 1)) * pw;

      vline.setAttribute("x1", px.toFixed(1));
      vline.setAttribute("x2", px.toFixed(1));
      vline.style.display = "block";

      tooltipEl.innerHTML = renderTipHtml(idx);
      tooltipEl.style.display = "block";
      const leftPos = Math.min(rect.width - 210, Math.max(8, evt.clientX - rect.left + 14));
      tooltipEl.style.left = `${leftPos}px`;
      tooltipEl.style.top = "10px";
    });

    overlay.addEventListener("mouseleave", () => {
      vline.style.display = "none";
      tooltipEl.style.display = "none";
    });
  }

  /**
   * 60-day Price Action + 22-day Forward ML Quantile Cone (P10-P90, P25-P75, P50 Median)
   */
  function renderFanChart(svgEl, tooltipEl, closes60, dates60, quantilesPct, horizonDays, tickerName) {
    const p = pal();
    const W = 740, H = 225;
    const pad = { l: 64, r: 22, t: 30, b: 30 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const lastPx = closes60[closes60.length - 1];
    // Quantiles indices out of 19 (0.05..0.95): P10=1, P25=4, P50=9, P75=14, P90=17
    const q10 = quantilesPct[1] / 100.0;
    const q25 = quantilesPct[4] / 100.0;
    const q50 = quantilesPct[9] / 100.0;
    const q75 = quantilesPct[14] / 100.0;
    const q90 = quantilesPct[17] / 100.0;

    const stepsF = horizonDays;
    const fan = [];
    for (let k = 0; k <= stepsF; k++) {
      const f = k / stepsF;
      const sq = Math.sqrt(f);
      fan.push({
        label: k === 0 ? dates60[dates60.length - 1] : `T+${k}`,
        p10: lastPx * (1.0 + q10 * sq),
        p25: lastPx * (1.0 + q25 * sq),
        p50: lastPx * (1.0 + q50 * f),
        p75: lastPx * (1.0 + q75 * sq),
        p90: lastPx * (1.0 + q90 * sq),
      });
    }

    const totalLen = closes60.length + stepsF;
    const allPrices = [...closes60, ...fan.map((x) => x.p10), ...fan.map((x) => x.p90)];
    const yMin = Math.min(...allPrices) * 0.985;
    const yMax = Math.max(...allPrices) * 1.015;
    const sx = (i) => pad.l + (i / (totalLen - 1)) * pw;
    const sy = (v) => pad.t + ph - ((v - yMin) / (yMax - yMin || 1)) * ph;

    const parts = [];
    for (let k = 0; k <= 4; k++) {
      const yv = yMin + (k / 4) * (yMax - yMin);
      const py = sy(yv);
      parts.push(`<line x1="${pad.l}" y1="${py.toFixed(1)}" x2="${W - pad.r}" y2="${py.toFixed(1)}" stroke="${p.grid}" stroke-width="1" />`);
      parts.push(`<text x="${pad.l - 8}" y="${(py + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="${p.muted}" font-family="var(--font-mono)">${yv.toFixed(1)}</text>`);
    }

    const splitX = sx(closes60.length - 1);
    parts.push(`<line x1="${splitX.toFixed(1)}" y1="${pad.t}" x2="${splitX.toFixed(1)}" y2="${H - pad.b}" stroke="${p.border}" stroke-dasharray="4 3" stroke-width="1" />`);
    parts.push(`<text x="${(splitX + 6).toFixed(1)}" y="${pad.t + 12}" font-size="10.5" fill="${p.muted}">Dự báo ML +${horizonDays} phiên</text>`);

    // Outer fan P10..P90
    const startI = closes60.length - 1;
    const top90 = fan.map((pt, k) => `${k === 0 ? "M" : "L"}${sx(startI + k).toFixed(1)},${sy(pt.p90).toFixed(1)}`).join(" ");
    const bot10 = fan.slice().reverse().map((pt, k) => `L${sx(startI + (fan.length - 1 - k)).toFixed(1)},${sy(pt.p10).toFixed(1)}`).join(" ");
    parts.push(`<path d="${top90} ${bot10} Z" fill="${p.bench}" opacity="0.16" />`);

    // Inner fan P25..P75
    const top75 = fan.map((pt, k) => `${k === 0 ? "M" : "L"}${sx(startI + k).toFixed(1)},${sy(pt.p75).toFixed(1)}`).join(" ");
    const bot25 = fan.slice().reverse().map((pt, k) => `L${sx(startI + (fan.length - 1 - k)).toFixed(1)},${sy(pt.p25).toFixed(1)}`).join(" ");
    parts.push(`<path d="${top75} ${bot25} Z" fill="${p.bull}" opacity="0.24" />`);

    // Historical close line
    const histD = closes60.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
    parts.push(`<path d="${histD}" fill="none" stroke="${p.ink}" stroke-width="2" />`);

    // Median projection line P50
    const medD = fan.map((pt, k) => `${k === 0 ? "M" : "L"}${sx(startI + k).toFixed(1)},${sy(pt.p50).toFixed(1)}`).join(" ");
    parts.push(`<path d="${medD}" fill="none" stroke="${p.bull}" stroke-width="2.2" stroke-dasharray="5 3" />`);

    parts.push(`<text x="${pad.l}" y="18" font-size="12" font-weight="700" fill="${p.ink}">Giá lịch sử 60 phiên &amp; Hình nón Phân vị Tương lai (P10–P90): ${tickerName}</text>`);
    parts.push(`<text x="${pad.l}" y="${H - 6}" font-size="10.5" fill="${p.muted}">${dates60[0]}</text>`);
    parts.push(`<text x="${splitX.toFixed(1)}" y="${H - 6}" text-anchor="middle" font-size="10.5" fill="${p.muted}">${dates60[dates60.length - 1]}</text>`);
    parts.push(`<text x="${W - pad.r}" y="${H - 6}" text-anchor="end" font-size="10.5" fill="${p.bull}">T+${horizonDays} (P50: ${fan[fan.length - 1].p50.toFixed(1)})</text>`);

    svgEl.innerHTML = parts.join("");

    const dummyX = new Array(totalLen).fill(0);
    attachCrosshair(svgEl, tooltipEl, dummyX, pad, W, H, (idx) => {
      if (idx < closes60.length) {
        return `<strong>${dates60[idx]}</strong><br>Giá đóng cửa: ${fmt(closes60[idx], 2)}`;
      }
      const k = idx - (closes60.length - 1);
      const pt = fan[Math.min(fan.length - 1, k)];
      return `<strong>Dự báo ${pt.label}</strong><br>P90: ${fmt(pt.p90, 1)} | P50: ${fmt(pt.p50, 1)} | P10: ${fmt(pt.p10, 1)}`;
    });
  }

  function renderDualCdfCharts(svgCdf, tipCdf, svgF2, tipF2, gridPct, cdfA, cdfB, f2A, f2B, nameA, nameB) {
    const p = pal();

    const drawSingle = (svgEl, tipEl, valsA, valsB, title, yLabel, shadeFlag, isF2) => {
      const W = 520, H = 220;
      const pad = { l: 66, r: 22, t: 28, b: 36 };
      const pw = W - pad.l - pad.r;
      const ph = H - pad.t - pad.b;

      const xMin = gridPct[0], xMax = gridPct[gridPct.length - 1];
      const yMin = 0.0;
      const yMax = isF2 ? Math.max(...valsA, ...valsB, 0.01) * 1.1 : 1.00;
      const sx = (v) => pad.l + ((v - xMin) / (xMax - xMin || 1)) * pw;
      const sy = (v) => pad.t + ph - ((v - yMin) / (yMax - yMin || 1)) * ph;

      const parts = [];
      for (let k = 0; k <= 4; k++) {
        const yv = yMin + (k / 4) * (yMax - yMin);
        const py = sy(yv);
        parts.push(`<line x1="${pad.l}" y1="${py.toFixed(1)}" x2="${W - pad.r}" y2="${py.toFixed(1)}" stroke="${p.grid}" stroke-width="1" />`);
        parts.push(`<text x="${pad.l - 8}" y="${(py + 4).toFixed(1)}" text-anchor="end" font-size="10.5" fill="${p.muted}" font-family="var(--font-mono)">${yv.toFixed(isF2 ? 3 : 2)}</text>`);
      }

      for (let k = 0; k <= 4; k++) {
        const xv = xMin + (k / 4) * (xMax - xMin);
        const px = sx(xv);
        const anchor = k === 0 ? "start" : k === 4 ? "end" : "middle";
        parts.push(`<line x1="${px.toFixed(1)}" y1="${pad.t}" x2="${px.toFixed(1)}" y2="${H - pad.b}" stroke="${p.grid}" stroke-width="1" />`);
        parts.push(`<text x="${px.toFixed(1)}" y="${H - pad.b + 15}" text-anchor="${anchor}" font-size="10.5" fill="${p.muted}" font-family="var(--font-mono)">${xv.toFixed(0)}%</text>`);
      }

      const xz = sx(0);
      parts.push(`<line x1="${xz.toFixed(1)}" y1="${pad.t}" x2="${xz.toFixed(1)}" y2="${H - pad.b}" stroke="${p.border}" stroke-dasharray="3 3" stroke-width="1" />`);

      // Shading violation region where Stock A CDF / F2 > Bench B
      if (shadeFlag) {
        let seg = [];
        const flush = () => {
          if (seg.length < 2) return;
          const top = seg.map((pt, i) => `${i === 0 ? "M" : "L"}${pt.x.toFixed(1)},${pt.yt.toFixed(1)}`).join(" ");
          const bot = seg.slice().reverse().map((pt) => `L${pt.x.toFixed(1)},${pt.yb.toFixed(1)}`).join(" ");
          parts.push(`<path d="${top} ${bot} Z" fill="${p.bear}" opacity="0.34" />`);
        };
        for (let i = 0; i < gridPct.length; i++) {
          if (valsA[i] > valsB[i]) {
            seg.push({ x: sx(gridPct[i]), yt: sy(valsA[i]), yb: sy(valsB[i]) });
          } else if (seg.length) {
            flush();
            seg = [];
          }
        }
        flush();
      }

      const pathB = gridPct.map((xv, i) => `${i === 0 ? "M" : "L"}${sx(xv).toFixed(1)},${sy(valsB[i]).toFixed(1)}`).join(" ");
      const pathA = gridPct.map((xv, i) => `${i === 0 ? "M" : "L"}${sx(xv).toFixed(1)},${sy(valsA[i]).toFixed(1)}`).join(" ");
      parts.push(`<path d="${pathB}" fill="none" stroke="${p.bench}" stroke-width="2" stroke-dasharray="5 3" />`);
      parts.push(`<path d="${pathA}" fill="none" stroke="${p.bull}" stroke-width="2.5" />`);

      parts.push(`<text x="${pad.l}" y="17" font-size="11.5" font-weight="700" fill="${p.ink}">${title}</text>`);
      parts.push(`<text x="14" y="${(pad.t + (H - pad.b)) / 2}" text-anchor="middle" font-size="10.5" fill="${p.muted}" transform="rotate(-90 14 ${(pad.t + (H - pad.b)) / 2})">${yLabel}</text>`);
      parts.push(`<text x="${(pad.l + W - pad.r) / 2}" y="${H - 4}" text-anchor="middle" font-size="10.5" fill="${p.muted}">Lợi suất ngưỡng t (%)</text>`);

      svgEl.innerHTML = parts.join("");
      attachCrosshair(svgEl, tipEl, gridPct, pad, W, H, (idx) => {
        const diff = valsA[idx] - valsB[idx];
        return `<strong>Ngưỡng r = ${fmt(gridPct[idx], 1)}%</strong><br>${nameA}: ${fmt(valsA[idx], isF2 ? 4 : 3)}<br>${nameB}: ${fmt(valsB[idx], isF2 ? 4 : 3)}<br>Chênh (A-B): <span class="${diff <= 0 ? "pos" : "neg"}">${diff >= 0 ? "+" : ""}${fmt(diff, isF2 ? 4 : 3)}</span>`;
      });
    };

    drawSingle(svgCdf, tipCdf, cdfA, cdfB, `1. CDF Dự báo F(r): ${nameA} vs ${nameB}`, "F(r)", true, false);
    drawSingle(svgF2, tipF2, f2A, f2B, `2. Kiểm định Almost SSD trên F(2)(t): Vùng đỏ = S2`, "F(2)(t)", true, true);
  }

  function renderMultiEquityAndDrawdown(svgEq, tipEq, svgDd, btData, seriesConfig) {
    const p = pal();
    const colorMap = {
      bull: p.bull,
      teal: p.teal,
      bench: p.bench,
      amber: p.amber,
      bear: p.bear,
      violet: p.violet,
      muted: p.muted,
    };
    const dates = btData.dates;
    const xIdx = dates.map((_, i) => i);

    const active = seriesConfig.filter((s) => s.visible && btData.curves[s.key]);
    if (!active.length) return;

    // 1) Equity Chart
    const W = 780, H = 290;
    const pad = { l: 74, r: 24, t: 30, b: 34 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const allY = active.flatMap((s) => btData.curves[s.key]);
    const yMin = Math.min(...allY) * 0.96;
    const yMax = Math.max(...allY) * 1.04;
    const sx = (i) => pad.l + (i / Math.max(1, xIdx.length - 1)) * pw;
    const sy = (v) => pad.t + ph - ((v - yMin) / (yMax - yMin || 1)) * ph;

    const parts = [];
    for (let k = 0; k <= 4; k++) {
      const yv = yMin + (k / 4) * (yMax - yMin);
      const py = sy(yv);
      parts.push(`<line x1="${pad.l}" y1="${py.toFixed(1)}" x2="${W - pad.r}" y2="${py.toFixed(1)}" stroke="${p.grid}" stroke-width="1" />`);
      parts.push(`<text x="${pad.l - 10}" y="${(py + 4).toFixed(1)}" text-anchor="end" font-size="11" fill="${p.muted}" font-family="var(--font-mono)">${yv.toFixed(1)}</text>`);
    }
    for (let k = 0; k <= 4; k++) {
      const iTick = Math.round((k / 4) * (dates.length - 1));
      const px = sx(iTick);
      const anchor = k === 0 ? "start" : k === 4 ? "end" : "middle";
      parts.push(`<line x1="${px.toFixed(1)}" y1="${pad.t}" x2="${px.toFixed(1)}" y2="${H - pad.b}" stroke="${p.grid}" stroke-width="1" />`);
      parts.push(`<text x="${px.toFixed(1)}" y="${H - pad.b + 16}" text-anchor="${anchor}" font-size="10.5" fill="${p.muted}" font-family="var(--font-mono)">${dates[iTick]}</text>`);
    }

    for (const ser of active) {
      const vals = btData.curves[ser.key];
      const d = vals.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
      parts.push(`<path d="${d}" fill="none" stroke="${colorMap[ser.token] || p.ink}" stroke-width="${ser.width}" ${ser.dash ? `stroke-dasharray="${ser.dash}"` : ""} />`);
    }

    parts.push(`<text x="${pad.l}" y="18" font-size="12" font-weight="700" fill="${p.ink}">Tăng trưởng Vốn Out-of-Sample 45 Kỳ (2021–2026 · Khởi đầu = 100 triệu VND)</text>`);
    parts.push(`<text x="15" y="${(pad.t + (H - pad.b)) / 2}" text-anchor="middle" font-size="11" fill="${p.muted}" transform="rotate(-90 15 ${(pad.t + (H - pad.b)) / 2})">Giá trị (triệu VND)</text>`);

    svgEq.innerHTML = parts.join("");
    attachCrosshair(svgEq, tipEq, xIdx, pad, W, H, (idx) => {
      const lines = active.slice(0, 5).map((s) => `${s.short}: <strong>${fmt(btData.curves[s.key][idx], 2)}</strong>`);
      return `<strong>Kỳ ${idx} (${dates[idx]})</strong><br>${lines.join("<br>")}`;
    });

    // 2) Drawdown Underwater Chart
    const H2 = 140;
    const pad2 = { l: 74, r: 24, t: 24, b: 24 };
    const ph2 = H2 - pad2.t - pad2.b;
    const ddMap = {};
    let ddMin = -5.0;
    for (const ser of active) {
      let peak = -Infinity;
      ddMap[ser.key] = btData.curves[ser.key].map((v) => {
        if (v > peak) peak = v;
        const dd = (v / peak - 1.0) * 100.0;
        if (dd < ddMin) ddMin = dd;
        return dd;
      });
    }
    ddMin = Math.floor(ddMin * 1.15);
    const sy2 = (v) => pad2.t + ph2 - ((v - ddMin) / (0 - ddMin || 1)) * ph2;
    const parts2 = [];
    for (let k = 0; k <= 2; k++) {
      const yv = ddMin + (k / 2) * (0 - ddMin);
      const py = sy2(yv);
      parts2.push(`<line x1="${pad2.l}" y1="${py.toFixed(1)}" x2="${W - pad2.r}" y2="${py.toFixed(1)}" stroke="${p.grid}" stroke-width="1" />`);
      parts2.push(`<text x="${pad2.l - 10}" y="${(py + 4).toFixed(1)}" text-anchor="end" font-size="10.5" fill="${p.muted}" font-family="var(--font-mono)">${yv.toFixed(0)}%</text>`);
    }
    for (const ser of active) {
      const vals = ddMap[ser.key];
      const d = vals.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${sy2(v).toFixed(1)}`).join(" ");
      parts2.push(`<path d="${d}" fill="none" stroke="${colorMap[ser.token] || p.ink}" stroke-width="${ser.width}" ${ser.dash ? `stroke-dasharray="${ser.dash}"` : ""} />`);
    }
    parts2.push(`<text x="${pad2.l}" y="15" font-size="11" font-weight="700" fill="${p.ink}">Biểu đồ Sụt giảm Vốn (Underwater Drawdown %)</text>`);
    svgDd.innerHTML = parts2.join("");
  }

  function renderSimpleMultiChart(svgEl, tipEl, xLabels, seriesList, title, yLabel, isBar = false) {
    const p = pal();
    const W = 580, H = 225;
    const pad = { l: 64, r: 18, t: 28, b: 32 };
    const pw = W - pad.l - pad.r;
    const ph = H - pad.t - pad.b;

    const allY = seriesList.flatMap((s) => s.vals);
    const yMin = isBar ? 0 : Math.min(...allY) * 0.96;
    const yMax = Math.max(...allY, 0.01) * (isBar ? 1.15 : 1.04);
    const sx = (i) => pad.l + (i / Math.max(1, xLabels.length - 1)) * pw;
    const sy = (v) => pad.t + ph - ((v - yMin) / (yMax - yMin || 1)) * ph;

    const parts = [];
    for (let k = 0; k <= 4; k++) {
      const yv = yMin + (k / 4) * (yMax - yMin);
      const py = sy(yv);
      parts.push(`<line x1="${pad.l}" y1="${py.toFixed(1)}" x2="${W - pad.r}" y2="${py.toFixed(1)}" stroke="${p.grid}" stroke-width="1" />`);
      parts.push(`<text x="${pad.l - 8}" y="${(py + 4).toFixed(1)}" text-anchor="end" font-size="10.5" fill="${p.muted}" font-family="var(--font-mono)">${yv.toFixed(isBar ? 2 : 1)}</text>`);
    }

    if (isBar) {
      const slotW = pw / xLabels.length;
      const bw = Math.max(6, (slotW * 0.72) / seriesList.length);
      for (let i = 0; i < xLabels.length; i++) {
        const cx = pad.l + (i + 0.5) * slotW;
        parts.push(`<text x="${cx.toFixed(1)}" y="${H - pad.b + 15}" text-anchor="middle" font-size="10.5" fill="${p.muted}" font-family="var(--font-mono)">${xLabels[i]}</text>`);
        seriesList.forEach((ser, si) => {
          const x = cx - (seriesList.length * bw) / 2 + si * bw;
          const topY = sy(ser.vals[i]);
          const hBar = Math.max(1, H - pad.b - topY);
          parts.push(`<rect x="${x.toFixed(1)}" y="${topY.toFixed(1)}" width="${(bw - 2).toFixed(1)}" height="${hBar.toFixed(1)}" fill="${p[ser.token] || p.bench}" opacity="0.88" rx="2" />`);
        });
      }
    } else {
      for (let k = 0; k <= 3; k++) {
        const idx = Math.round((k / 3) * (xLabels.length - 1));
        const anchor = k === 0 ? "start" : k === 3 ? "end" : "middle";
        parts.push(`<text x="${sx(idx).toFixed(1)}" y="${H - pad.b + 15}" text-anchor="${anchor}" font-size="10.5" fill="${p.muted}" font-family="var(--font-mono)">${xLabels[idx]}</text>`);
      }
      for (const ser of seriesList) {
        const d = ser.vals.map((v, i) => `${i === 0 ? "M" : "L"}${sx(i).toFixed(1)},${sy(v).toFixed(1)}`).join(" ");
        parts.push(`<path d="${d}" fill="none" stroke="${p[ser.token] || p.bull}" stroke-width="${ser.width || 2.3}" ${ser.dash ? `stroke-dasharray="${ser.dash}"` : ""} />`);
      }
    }

    parts.push(`<text x="${pad.l}" y="17" font-size="11.5" font-weight="700" fill="${p.ink}">${title}</text>`);
    parts.push(`<text x="14" y="${(pad.t + (H - pad.b)) / 2}" text-anchor="middle" font-size="10.5" fill="${p.muted}" transform="rotate(-90 14 ${(pad.t + (H - pad.b)) / 2})">${yLabel}</text>`);
    svgEl.innerHTML = parts.join("");

    if (tipEl) {
      attachCrosshair(svgEl, tipEl, xLabels, pad, W, H, (idx) => {
        const rows = seriesList.map((s) => `${s.label}: <strong>${fmt(s.vals[idx], 2)}</strong>`).join("<br>");
        return `<strong>${xLabels[idx]}</strong><br>${rows}`;
      });
    }
  }

  return {
    pal,
    renderFanChart,
    renderDualCdfCharts,
    renderMultiEquityAndDrawdown,
    renderSimpleMultiChart,
  };
})();
