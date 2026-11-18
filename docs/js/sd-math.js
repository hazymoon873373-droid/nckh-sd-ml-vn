/**
 * sd-math.js — Client-side Stochastic Dominance (FSD / SSD / ASSD Tzeng et al., 2013)
 * and Projected-Simplex Portfolio Optimizer (Cesarone & Puerto, 2025; Luo et al., 2025).
 */
window.SDMath = (() => {
  "use strict";

  function quantilesToCdf(quantiles, taus, grid) {
    const q = [...quantiles].sort((a, b) => a - b);
    const spread = Math.max(q[q.length - 1] - q[0], 4.0); // grid is in pct (-32 to +40)
    const kx = [q[0] - 0.18 * spread, ...q, q[q.length - 1] + 0.18 * spread];
    const ky = [0.0, ...taus, 1.0];

    const out = new Array(grid.length);
    let j = 0;
    let prev = 0;
    for (let i = 0; i < grid.length; i++) {
      const x = grid[i];
      let val;
      if (x <= kx[0]) {
        val = 0.0;
      } else if (x >= kx[kx.length - 1]) {
        val = 1.0;
      } else {
        while (j < kx.length - 2 && kx[j + 1] < x) j++;
        const dx = kx[j + 1] - kx[j];
        const t = dx > 1e-9 ? (x - kx[j]) / dx : 0.0;
        val = ky[j] + t * (ky[j + 1] - ky[j]);
      }
      prev = Math.max(prev, val);
      out[i] = Math.min(1.0, Math.max(0.0, prev));
    }
    return out;
  }

  function computeSdMetrics(cdfA, cdfB, gridPct) {
    // gridPct is in percentage points (-32..+40). Convert dx to decimal return units
    const dx = (gridPct[1] - gridPct[0]) / 100.0;
    const x0 = gridPct[0] / 100.0;

    let meanA = x0;
    let meanB = x0;
    let areaViol1 = 0;
    let areaTot1 = 0;

    const f2A = new Array(gridPct.length);
    const f2B = new Array(gridPct.length);
    let cumA = 0;
    let cumB = 0;

    for (let i = 0; i < gridPct.length; i++) {
      meanA += (1.0 - cdfA[i]) * dx;
      meanB += (1.0 - cdfB[i]) * dx;
      const d1 = cdfA[i] - cdfB[i];
      if (d1 > 0) areaViol1 += d1 * dx;
      areaTot1 += Math.abs(d1) * dx;

      cumA += cdfA[i] * dx;
      cumB += cdfB[i] * dx;
      f2A[i] = cumA;
      f2B[i] = cumB;
    }

    let areaViol2 = 0;
    let areaRev2 = 0;
    let areaTot2 = 0;
    for (let i = 0; i < gridPct.length; i++) {
      const d2 = f2A[i] - f2B[i];
      if (d2 > 0) areaViol2 += d2 * dx;
      else areaRev2 += -d2 * dx;
      areaTot2 += Math.abs(d2) * dx;
    }

    const eps1 = areaTot1 > 1e-12 ? areaViol1 / areaTot1 : 0.0;
    const eps2 = areaTot2 > 1e-12 ? areaViol2 / areaTot2 : 0.0;
    const eps2Rev = areaTot2 > 1e-12 ? areaRev2 / areaTot2 : 0.0;
    const meanOk = meanA >= meanB - 1e-6;

    return {
      fsd_strict: eps1 < 1e-6 && areaTot1 > 1e-8,
      ssd_strict: meanOk && eps2 < 1e-6 && areaTot2 > 1e-8,
      eps1_star: eps1,
      eps2_star: eps2,
      eps2_reverse: eps2Rev,
      mean_a: meanA,
      mean_b: meanB,
      mean_ok: meanOk,
      area_viol2: areaViol2,
      area_total2: areaTot2,
      f2_a: f2A,
      f2_b: f2B,
    };
  }

  function classifySd(sd, epsThreshold) {
    if (sd.ssd_strict) return { label: "YÊU (SSD Chuẩn)", cls: "badge-bull", isLoved: true };
    if (sd.mean_ok && sd.eps2_star <= epsThreshold) return { label: "YÊU (Almost SD)", cls: "badge-bull", isLoved: true };
    if (!sd.mean_ok && sd.eps2_reverse <= epsThreshold) return { label: "KHÔNG YÊU (Bị chi phối)", cls: "badge-bear", isLoved: false };
    return { label: "TRUNG TÍNH", cls: "badge-neutral", isLoved: false };
  }

  function projectCappedSimplex(vec, uMax) {
    const n = vec.length;
    if (uMax * n < 1.0 - 1e-9) return new Array(n).fill(1.0 / n);
    let lo = Math.min(...vec) - 1.0;
    let hi = Math.max(...vec);
    for (let it = 0; it < 45; it++) {
      const mid = 0.5 * (lo + hi);
      let s = 0;
      for (let i = 0; i < n; i++) s += Math.min(uMax, Math.max(0.0, vec[i] - mid));
      if (s > 1.0) lo = mid;
      else hi = mid;
    }
    const mid = 0.5 * (lo + hi);
    const w = new Array(n);
    let s = 0;
    for (let i = 0; i < n; i++) {
      w[i] = Math.min(uMax, Math.max(0.0, vec[i] - mid));
      s += w[i];
    }
    return s > 1e-12 ? w.map((x) => x / s) : new Array(n).fill(1.0 / n);
  }

  function interp1d(x, xs, ys) {
    if (x <= xs[0]) return ys[0];
    if (x >= xs[xs.length - 1]) return ys[ys.length - 1];
    let lo = 0;
    let hi = xs.length - 1;
    while (hi - lo > 1) {
      const m = (lo + hi) >> 1;
      if (xs[m] <= x) lo = m;
      else hi = m;
    }
    const span = xs[hi] - xs[lo];
    const t = span > 1e-9 ? (x - xs[lo]) / span : 0.0;
    return ys[lo] + t * (ys[hi] - ys[lo]);
  }

  /**
   * Live Scenario-based ASSD Portfolio Optimization directly in browser
   * (Cesarone & Puerto, 2025 EJOR; Luo et al., 2025 MS).
   */
  function optimizeAssdPortfolio(predQuantilesDec, tausDec, histDailyDec, prevW, opts = {}) {
    const wMax = opts.wMax || 0.20;
    const turnPenalty = opts.turnPenalty !== undefined ? opts.turnPenalty : 0.012;
    const meanBonus = opts.meanBonus !== undefined ? opts.meanBonus : 0.28;
    const lotteryPenaltyArr = opts.lotteryPenalty || null;

    const nStocks = predQuantilesDec.length;
    const nHist = histDailyDec.length;
    const mRows = Math.max(30, Math.min(90, nHist));
    const startRow = nHist - mRows;

    // Build correlated joint scenarios via trailing empirical ranks -> predicted ML quantiles
    const scen = Array.from({ length: mRows }, () => new Float64Array(nStocks));
    for (let j = 0; j < nStocks; j++) {
      const pairs = [];
      for (let t = 0; t < mRows; t++) pairs.push({ val: histDailyDec[startRow + t][j], idx: t });
      pairs.sort((a, b) => a.val - b.val);
      for (let r = 0; r < mRows; r++) {
        const u = (r + 0.5) / mRows;
        scen[pairs[r].idx][j] = interp1d(u, tausDec, predQuantilesDec[j]);
      }
    }

    const benchScen = new Float64Array(mRows);
    const stockMeans = new Float64Array(nStocks);
    for (let t = 0; t < mRows; t++) {
      let rowSum = 0;
      for (let j = 0; j < nStocks; j++) {
        rowSum += scen[t][j];
        stockMeans[j] += scen[t][j] / mRows;
      }
      benchScen[t] = rowSum / nStocks;
    }

    const benchSorted = Array.from(benchScen).sort((a, b) => a - b);
    const nThr = 15;
    const thresholds = new Float64Array(nThr);
    const targetF2 = new Float64Array(nThr);
    for (let k = 0; k < nThr; k++) {
      const qIdx = Math.min(mRows - 1, Math.floor((0.05 + (0.80 * k) / (nThr - 1)) * mRows));
      thresholds[k] = benchSorted[qIdx];
      let f2Sum = 0;
      for (let t = 0; t < mRows; t++) {
        const d = thresholds[k] - benchScen[t];
        if (d > 0) f2Sum += d;
      }
      targetF2[k] = f2Sum / mRows;
    }

    let w = prevW && prevW.reduce((a, b) => a + b, 0) > 0.5
      ? [...prevW]
      : new Array(nStocks).fill(1.0 / nStocks);

    const rp = new Float64Array(mRows);
    const grad = new Array(nStocks);
    for (let it = 0; it < 70; it++) {
      for (let t = 0; t < mRows; t++) {
        let dot = 0;
        for (let j = 0; j < nStocks; j++) dot += scen[t][j] * w[j];
        rp[t] = dot;
      }
      grad.fill(0.0);
      for (let k = 0; k < nThr; k++) {
        const thr = thresholds[k];
        let portF2 = 0;
        for (let t = 0; t < mRows; t++) {
          if (thr > rp[t]) portF2 += (thr - rp[t]);
        }
        portF2 /= mRows;
        if (portF2 > targetF2[k]) {
          const excess = portF2 - targetF2[k];
          for (let j = 0; j < nStocks; j++) {
            let dk = 0;
            for (let t = 0; t < mRows; t++) {
              if (thr > rp[t]) dk -= scen[t][j];
            }
            grad[j] += 2.0 * excess * (dk / mRows) * 120.0;
          }
        }
      }
      const step = 0.65 / (1.0 + 0.045 * it);
      const nextVec = new Array(nStocks);
      for (let j = 0; j < nStocks; j++) {
        const prevTerm = prevW ? 2.0 * turnPenalty * (w[j] - prevW[j]) : 0.0;
        const lotTerm = lotteryPenaltyArr ? lotteryPenaltyArr[j] : 0.0;
        nextVec[j] = w[j] - step * (grad[j] - meanBonus * stockMeans[j] + prevTerm + lotTerm);
      }
      w = projectCappedSimplex(nextVec, wMax);
    }
    return w;
  }

  function computePortfolioStats(retsDec, turnovers, benchDec, horizonDays) {
    const n = retsDec.length;
    if (!n) return {};
    let eq = 1.0;
    const eqArr = [1.0];
    let peak = 1.0;
    let mdd = 0.0;
    let gains = 0.0;
    let losses = 0.0;
    let negSqSum = 0.0;
    let negCount = 0;
    let sumR = 0.0;
    let beatCount = 0;

    for (let i = 0; i < n; i++) {
      const r = retsDec[i];
      eq *= (1.0 + r);
      eqArr.push(eq);
      if (eq > peak) peak = eq;
      const dd = eq / peak - 1.0;
      if (dd < mdd) mdd = dd;
      if (r >= 0) gains += r;
      else {
        losses += -r;
        negSqSum += r * r;
        negCount++;
      }
      sumR += r;
      if (r > benchDec[i]) beatCount++;
    }

    const meanR = sumR / n;
    let varSum = 0;
    for (let i = 0; i < n; i++) varSum += (retsDec[i] - meanR) ** 2;
    const stdR = Math.sqrt(varSum / Math.max(n - 1, 1));
    const periodsYr = 252.0 / horizonDays;

    const cumRet = (eq - 1.0) * 100.0;
    const annRet = (Math.pow(Math.max(eq, 0.01), periodsYr / n) - 1.0) * 100.0;
    const annVol = stdR * Math.sqrt(periodsYr) * 100.0;
    const sharpe = stdR > 1e-8 ? (meanR / stdR) * Math.sqrt(periodsYr) : 0.0;
    const downDev = negCount > 0 ? Math.sqrt(negSqSum / negCount) : Math.max(stdR, 1e-8);
    const sortino = (meanR / downDev) * Math.sqrt(periodsYr);
    const omega = losses > 1e-8 ? gains / losses : 3.0;
    const avgTurn = turnovers.length ? (turnovers.reduce((a, b) => a + b, 0) / turnovers.length) * 100.0 : 0.0;

    return {
      cum_return_pct: cumRet,
      ann_return_pct: annRet,
      ann_vol_pct: annVol,
      sharpe,
      sortino,
      omega,
      max_drawdown_pct: mdd * 100.0,
      win_vs_bench_pct: (beatCount / n) * 100.0,
      avg_turnover_pct: avgTurn,
    };
  }

  return {
    quantilesToCdf,
    computeSdMetrics,
    classifySd,
    optimizeAssdPortfolio,
    computePortfolioStats,
  };
})();
