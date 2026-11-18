/**
 * ml-engine.js — Genuine In-Browser Distributional ML Engine
 * Runs Two-Stage Vol-Scaled Quantile Ridge (IRLS), In-Browser Quantile Decision Forest,
 * Cross-Sectional Feature Ranking, Walk-Forward OOS Evaluation, and Custom CSV Analysis
 * directly on client-side typed arrays.
 */
window.BrowserML = (() => {
  "use strict";

  const TAUS_LIVE = [0.05, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90, 0.95];

  function solveLinearSystem(A, b) {
    const n = A.length;
    const M = A.map((row, i) => [...row, b[i]]);
    for (let col = 0; col < n; col++) {
      let piv = col;
      for (let r = col + 1; r < n; r++) {
        if (Math.abs(M[r][col]) > Math.abs(M[piv][col])) piv = r;
      }
      if (Math.abs(M[piv][col]) < 1e-12) continue;
      [M[col], M[piv]] = [M[piv], M[col]];
      const div = M[col][col];
      for (let j = col; j <= n; j++) M[col][j] /= div;
      for (let r = 0; r < n; r++) {
        if (r === col) continue;
        const factor = M[r][col];
        for (let j = col; j <= n; j++) M[r][j] -= factor * M[col][j];
      }
    }
    return M.map((row) => row[n]);
  }

  function quantileFromSorted(sortedArr, tau) {
    if (!sortedArr.length) return 0;
    const idx = tau * (sortedArr.length - 1);
    const lo = Math.floor(idx);
    const hi = Math.ceil(idx);
    const frac = idx - lo;
    return sortedArr[lo] + frac * (sortedArr[hi] - sortedArr[lo]);
  }

  function dotVec(w, x) {
    let s = w[0];
    for (let j = 0; j < x.length; j++) s += w[j + 1] * x[j];
    return s;
  }

  function buildCrossSectionalFeatures(panel, horizon) {
    const T = panel.dates.length;
    const N = panel.tickers.length;
    const nFeat = 8; // Compact high-signal browser feature vector matching 18-feature families
    const featRaw = Array.from({ length: T }, () => Array.from({ length: N }, () => new Float64Array(nFeat)));
    const featRank = Array.from({ length: T }, () => Array.from({ length: N }, () => new Float64Array(nFeat)));
    const ewmaVol = Array.from({ length: T }, () => new Float64Array(N));
    const fwdRet = Array.from({ length: T }, () => new Float64Array(N));

    const sqH = Math.sqrt(horizon);

    for (let j = 0; j < N; j++) {
      let ewVar = 0.0004;
      for (let t = 0; t < T; t++) {
        const r1 = panel.rets[t][j] / 100.0;
        ewVar = 0.06 * (r1 * r1) + 0.94 * ewVar;
        const sig = Math.max(Math.sqrt(ewVar) * sqH, 0.015);
        ewmaVol[t][j] = sig;

        const cNow = panel.closes[t][j];
        const c5 = panel.closes[Math.max(0, t - 5)][j];
        const c22 = panel.closes[Math.max(0, t - 22)][j];
        const c66 = panel.closes[Math.max(0, t - 66)][j];

        let max22 = -1;
        let downSq = 0;
        for (let k = Math.max(0, t - 21); k <= t; k++) {
          const rk = panel.rets[k][j] / 100.0;
          if (rk > max22) max22 = rk;
          if (rk < 0) downSq += rk * rk;
        }
        const semi22 = Math.sqrt(downSq / 22.0) * sqH;

        const vec = featRaw[t][j];
        vec[0] = r1;
        vec[1] = c5 > 0 ? cNow / c5 - 1.0 : 0.0;
        vec[2] = c22 > 0 ? cNow / c22 - 1.0 : 0.0;
        vec[3] = c66 > 0 ? cNow / c66 - 1.0 : 0.0;
        vec[4] = sig;
        vec[5] = max22 > -1 ? max22 : 0.0; // Fong (2013) lottery proxy
        vec[6] = semi22;
        vec[7] = panel.cmf[t][j];

        if (t + horizon < T && cNow > 0) {
          fwdRet[t][j] = panel.closes[t + horizon][j] / cNow - 1.0;
        } else {
          fwdRet[t][j] = Number.NaN;
        }
      }
    }

    // Cross-sectional rank-standardize into [-1, 1] per date (Gu et al., 2020)
    for (let t = 0; t < T; t++) {
      for (let f = 0; f < nFeat; f++) {
        const order = Array.from({ length: N }, (_, j) => ({ v: featRaw[t][j][f], j })).sort((a, b) => a.v - b.v);
        for (let rank = 0; rank < N; rank++) {
          featRank[t][order[rank].j][f] = (rank / Math.max(N - 1, 1)) * 2.0 - 1.0;
        }
      }
    }

    return { T, N, nFeat, featRaw, featRank, ewmaVol, fwdRet };
  }

  function fitQuantileRidge(X, z, taus, l2 = 25.0) {
    const n = X.length;
    const p = X[0].length;
    const dim = p + 1;

    // Initial ridge solve
    const XtX = Array.from({ length: dim }, () => new Array(dim).fill(0));
    const Xty = new Array(dim).fill(0);
    for (let i = 0; i < n; i++) {
      const xi = X[i];
      for (let a = 0; a < dim; a++) {
        const va = a === 0 ? 1.0 : xi[a - 1];
        Xty[a] += va * z[i];
        for (let b = a; b < dim; b++) {
          const vb = b === 0 ? 1.0 : xi[b - 1];
          XtX[a][b] += va * vb;
        }
      }
    }
    for (let a = 0; a < dim; a++) {
      for (let b = 0; b < a; b++) XtX[a][b] = XtX[b][a];
      if (a > 0) XtX[a][a] += l2;
    }
    const betaInit = solveLinearSystem(XtX, Xty);

    const coefs = [];
    for (const tau of taus) {
      let beta = [...betaInit];
      for (let it = 0; it < 5; it++) {
        const A = Array.from({ length: dim }, () => new Array(dim).fill(0));
        const rhs = new Array(dim).fill(0);
        for (let i = 0; i < n; i++) {
          const r = z[i] - dotVec(beta, X[i]);
          const w = (r >= 0 ? tau : 1.0 - tau) / Math.max(Math.abs(r), 1e-3);
          const xi = X[i];
          for (let a = 0; a < dim; a++) {
            const va = a === 0 ? 1.0 : xi[a - 1];
            rhs[a] += w * va * z[i];
            for (let b = a; b < dim; b++) {
              const vb = b === 0 ? 1.0 : xi[b - 1];
              A[a][b] += w * va * vb;
            }
          }
        }
        for (let a = 0; a < dim; a++) {
          for (let b = 0; b < a; b++) A[a][b] = A[b][a];
          if (a > 0) A[a][a] += l2;
        }
        beta = solveLinearSystem(A, rhs);
      }
      coefs.push(beta);
    }
    return coefs;
  }

  function buildRandomQuantileForest(X, y, nTrees = 18, maxDepth = 4, minLeaf = 25) {
    const n = X.length;
    const p = X[0].length;
    let seed = 1337;
    const rand = () => {
      seed = (seed * 1664525 + 1013904223) >>> 0;
      return seed / 4294967296;
    };

    function variance(indices) {
      const m = indices.length;
      if (!m) return 0;
      let s = 0;
      for (const i of indices) s += y[i];
      const mu = s / m;
      let v = 0;
      for (const i of indices) v += (y[i] - mu) ** 2;
      return v;
    }

    function buildNode(indices, depth) {
      if (depth >= maxDepth || indices.length < 2 * minLeaf) {
        const vals = indices.map((i) => y[i]).sort((a, b) => a - b);
        return { leaf: true, vals };
      }
      let bestGain = 0;
      let bestF = -1;
      let bestThr = 0;
      const totVar = variance(indices);
      for (let t = 0; t < 4; t++) {
        const f = Math.floor(rand() * p);
        for (const thr of [-0.35, 0.0, 0.35]) {
          const left = [];
          const right = [];
          for (const i of indices) {
            if (X[i][f] <= thr) left.push(i);
            else right.push(i);
          }
          if (left.length < minLeaf || right.length < minLeaf) continue;
          const gain = totVar - (variance(left) + variance(right));
          if (gain > bestGain) {
            bestGain = gain;
            bestF = f;
            bestThr = thr;
          }
        }
      }
      if (bestF === -1) {
        return { leaf: true, vals: indices.map((i) => y[i]).sort((a, b) => a - b) };
      }
      const left = [];
      const right = [];
      for (const i of indices) {
        if (X[i][bestF] <= bestThr) left.push(i);
        else right.push(i);
      }
      return {
        leaf: false,
        f: bestF,
        thr: bestThr,
        left: buildNode(left, depth + 1),
        right: buildNode(right, depth + 1),
      };
    }

    const trees = [];
    const subSize = Math.min(n, 450);
    for (let k = 0; k < nTrees; k++) {
      const idx = [];
      for (let i = 0; i < subSize; i++) idx.push(Math.floor(rand() * n));
      trees.push(buildNode(idx, 0));
    }
    return trees;
  }

  function predictForestRow(trees, row, taus) {
    const pooled = [];
    for (const tr of trees) {
      let cur = tr;
      while (!cur.leaf) cur = row[cur.f] <= cur.thr ? cur.left : cur.right;
      for (const v of cur.vals) pooled.push(v);
    }
    pooled.sort((a, b) => a - b);
    return taus.map((tau) => quantileFromSorted(pooled, tau));
  }

  /**
   * Run live Walk-Forward Browser ML + ASSD Optimization over market_panel.json
   */
  function runLiveExperiment(panel, cfg) {
    const t0 = performance.now();
    const horizon = Number(cfg.horizon) || 22;
    const modelType = cfg.model || "ts_qml"; // ts_qml | qrf | hist
    const l2 = Number(cfg.l2) || 25.0;
    const wMax = Number(cfg.wMax) || 0.20;
    const txCost = (Number(cfg.txCostPct) || 0.35) / 100.0;
    const lotteryFilter = Boolean(cfg.lotteryFilter);
    const epsThr = Number(cfg.eps) || 0.059;

    const { T, N, featRaw, featRank, ewmaVol, fwdRet } = buildCrossSectionalFeatures(panel, horizon);
    const purge = horizon + 2;
    const startIdx = 140;
    const rebal = [];
    for (let d = startIdx; d < T - horizon - 1; d += horizon) rebal.push(d);

    let prevWAssd = new Array(N).fill(1.0 / N);
    const assdRets = [];
    const benchRets = [];
    const minVarRets = [];
    const turns = [];
    const curveDates = [panel.dates[rebal[0]]];

    const pinballTotals = new Array(TAUS_LIVE.length).fill(0);
    const pinballBench = new Array(TAUS_LIVE.length).fill(0);
    let evalCount = 0;

    // Pre-build daily returns matrix in decimals
    const dailyDec = panel.rets.map((row) => row.map((v) => v / 100.0));

    for (let s = 0; s < rebal.length; s++) {
      const rb = rebal[s];
      const trEnd = rb - purge;
      const X = [];
      const y = [];
      const sig = [];
      for (let d = 70; d < trEnd; d += 3) {
        for (let j = 0; j < N; j++) {
          const fr = fwdRet[d][j];
          if (!Number.isFinite(fr)) continue;
          X.push(featRank[d][j]);
          y.push(fr);
          sig.push(ewmaVol[d][j]);
        }
      }
      const z = y.map((yv, i) => Math.max(-4.5, Math.min(4.5, yv / sig[i])));

      const qPred = [];
      if (modelType === "ts_qml") {
        const coefs = fitQuantileRidge(X, z, TAUS_LIVE, l2);
        for (let j = 0; j < N; j++) {
          const rawQ = coefs.map((beta) => dotVec(beta, featRank[rb][j]) * ewmaVol[rb][j]);
          rawQ.sort((a, b) => a - b);
          qPred.push(rawQ);
        }
      } else if (modelType === "qrf") {
        const trees = buildRandomQuantileForest(X, y, 16, 4, 24);
        for (let j = 0; j < N; j++) {
          qPred.push(predictForestRow(trees, featRank[rb][j], TAUS_LIVE));
        }
      } else {
        // Historical empirical quantiles per stock
        for (let j = 0; j < N; j++) {
          const arr = [];
          for (let d = Math.max(0, trEnd - 100); d < trEnd; d++) {
            if (Number.isFinite(fwdRet[d][j])) arr.push(fwdRet[d][j]);
          }
          arr.sort((a, b) => a - b);
          qPred.push(arr.length >= 10 ? TAUS_LIVE.map((t) => quantileFromSorted(arr, t)) : TAUS_LIVE.map(() => 0.0));
        }
      }

      // Historical empirical benchmark quantiles for pinball baseline comparison
      for (let j = 0; j < N; j++) {
        const yt = fwdRet[rb][j];
        if (!Number.isFinite(yt)) continue;
        const arrHist = [];
        for (let d = Math.max(0, trEnd - 100); d < trEnd; d++) {
          if (Number.isFinite(fwdRet[d][j])) arrHist.push(fwdRet[d][j]);
        }
        arrHist.sort((a, b) => a - b);
        for (let k = 0; k < TAUS_LIVE.length; k++) {
          const tau = TAUS_LIVE[k];
          const eM = yt - qPred[j][k];
          pinballTotals[k] += Math.max(tau * eM, (tau - 1) * eM);
          const qH = arrHist.length >= 10 ? quantileFromSorted(arrHist, tau) : 0.0;
          const eH = yt - qH;
          pinballBench[k] += Math.max(tau * eH, (tau - 1) * eH);
        }
        evalCount++;
      }

      // Lottery penalty (Fong 2013): penalize top-25% MAX stocks if filter enabled
      let lotPen = null;
      if (lotteryFilter) {
        const maxVals = featRaw[rb].map((v) => v[5]).sort((a, b) => a - b);
        const maxCut = maxVals[Math.floor(0.75 * N)];
        lotPen = featRaw[rb].map((v) => (v[5] >= maxCut ? 0.35 : 0.0));
      }

      const histSlice = dailyDec.slice(Math.max(0, rb - 90), rb);
      const wNew = window.SDMath.optimizeAssdPortfolio(qPred, TAUS_LIVE, histSlice, prevWAssd, {
        wMax,
        turnPenalty: 0.012,
        meanBonus: 0.28,
        lotteryPenalty: lotPen,
      });

      // Inverse-vol defensive benchmark
      const invV = featRaw[rb].map((v) => 1.0 / Math.max(v[4], 0.02));
      const invSum = invV.reduce((a, b) => a + b, 0);
      const wInvV = invV.map((v) => v / invSum);

      let turn = 0;
      let rAssd = 0;
      let rBench = 0;
      let rDef = 0;
      for (let j = 0; j < N; j++) {
        turn += 0.5 * Math.abs(wNew[j] - prevWAssd[j]);
        const rj = fwdRet[rb][j] || 0;
        rAssd += wNew[j] * rj;
        rBench += rj / N;
        rDef += wInvV[j] * rj;
      }
      assdRets.push(rAssd - (s > 0 ? turn * txCost : 0.5 * txCost));
      benchRets.push(rBench);
      minVarRets.push(rDef - 0.15 * txCost);
      turns.push(turn);
      prevWAssd = wNew;
      curveDates.push(panel.dates[Math.min(rb + horizon, T - 1)]);
    }

    // Latest live forecast trained up to T - purge
    const Xfull = [];
    const yfull = [];
    const sfull = [];
    for (let d = 70; d < T - purge; d += 2) {
      for (let j = 0; j < N; j++) {
        if (!Number.isFinite(fwdRet[d][j])) continue;
        Xfull.push(featRank[d][j]);
        yfull.push(fwdRet[d][j]);
        sfull.push(ewmaVol[d][j]);
      }
    }
    const zfull = yfull.map((yv, i) => Math.max(-4.5, Math.min(4.5, yv / sfull[i])));
    const qNow = [];
    const lastIdx = T - 1;

    if (modelType === "ts_qml") {
      const coefs = fitQuantileRidge(Xfull, zfull, TAUS_LIVE, l2);
      for (let j = 0; j < N; j++) {
        const arr = coefs.map((beta) => dotVec(beta, featRank[lastIdx][j]) * ewmaVol[lastIdx][j]);
        arr.sort((a, b) => a - b);
        qNow.push(arr);
      }
    } else if (modelType === "qrf") {
      const trees = buildRandomQuantileForest(Xfull, yfull, 18, 4, 24);
      for (let j = 0; j < N; j++) qNow.push(predictForestRow(trees, featRank[lastIdx][j], TAUS_LIVE));
    } else {
      for (let j = 0; j < N; j++) {
        const arr = [];
        for (let d = Math.max(0, T - purge - 120); d < T - purge; d++) {
          if (Number.isFinite(fwdRet[d][j])) arr.push(fwdRet[d][j]);
        }
        arr.sort((a, b) => a - b);
        qNow.push(arr.length >= 10 ? TAUS_LIVE.map((t) => quantileFromSorted(arr, t)) : TAUS_LIVE.map(() => 0.0));
      }
    }

    let lotNow = null;
    if (lotteryFilter) {
      const maxVals = featRaw[lastIdx].map((v) => v[5]).sort((a, b) => a - b);
      const maxCut = maxVals[Math.floor(0.75 * N)];
      lotNow = featRaw[lastIdx].map((v) => (v[5] >= maxCut ? 0.35 : 0.0));
    }
    const wOptimalNow = window.SDMath.optimizeAssdPortfolio(qNow, TAUS_LIVE, dailyDec.slice(-90), prevWAssd, {
      wMax,
      turnPenalty: 0.012,
      meanBonus: 0.28,
      lotteryPenalty: lotNow,
    });

    // Compute ASSD scores vs live equal-weight predicted benchmark
    const liveGrid = [];
    for (let k = 0; k < 101; k++) liveGrid.push(-28.0 + k * 0.64);
    const liveCdfs = qNow.map((row) =>
      window.SDMath.quantilesToCdf(row.map((v) => v * 100.0), TAUS_LIVE, liveGrid)
    );
    const benchCdf = liveGrid.map((_, gi) => liveCdfs.reduce((acc, c) => acc + c[gi], 0) / N);

    const stockSummaries = panel.tickers.map((t, j) => {
      const sd = window.SDMath.computeSdMetrics(liveCdfs[j], benchCdf, liveGrid);
      const cls = window.SDMath.classifySd(sd, epsThr);
      return {
        ticker: t,
        sector: panel.sectors[j],
        pred_mean_pct: sd.mean_a * 100.0,
        q10_pct: qNow[j][1] * 100.0,
        q50_pct: qNow[j][4] * 100.0,
        q90_pct: qNow[j][7] * 100.0,
        eps2_star: sd.eps2_star,
        verdict: cls.label,
        verdict_cls: cls.cls,
        cmf_20: panel.cmf[lastIdx][j],
        weight_pct: wOptimalNow[j] * 100.0,
      };
    }).sort((a, b) => b.weight_pct - a.weight_pct || a.eps2_star - b.eps2_star);

    const toCurve = (arr) => {
      const out = [100.0];
      let v = 100.0;
      for (const r of arr) {
        v *= (1.0 + r);
        out.push(v);
      }
      return out;
    };

    const elapsedMs = Math.round(performance.now() - t0);
    const pinballByTau = TAUS_LIVE.map((tau, k) => ({
      tau: `${Math.round(tau * 100)}%`,
      model: evalCount ? (pinballTotals[k] / evalCount) * 100.0 : 0,
      hist: evalCount ? (pinballBench[k] / evalCount) * 100.0 : 0,
    }));

    const meanPinM = pinballByTau.reduce((a, b) => a + b.model, 0) / pinballByTau.length;
    const meanPinH = pinballByTau.reduce((a, b) => a + b.hist, 0) / pinballByTau.length;

    return {
      elapsedMs,
      horizon,
      nRebalances: rebal.length,
      evalCount,
      dates: curveDates,
      curves: {
        assd: toCurve(assdRets),
        invVol: toCurve(minVarRets),
        bench: toCurve(benchRets),
      },
      stats: {
        assd: window.SDMath.computePortfolioStats(assdRets, turns, benchRets, horizon),
        invVol: window.SDMath.computePortfolioStats(minVarRets, [], benchRets, horizon),
        bench: window.SDMath.computePortfolioStats(benchRets, [], benchRets, horizon),
      },
      pinballMeanPct: meanPinM,
      pinballHistPct: meanPinH,
      crpsPct: 2.0 * meanPinM,
      crpsHistPct: 2.0 * meanPinH,
      pinballByTau,
      stocks: stockSummaries,
    };
  }

  /**
   * Parse single-stock CSV (Date, Close, High, Low, Volume [, FNF]) uploaded by user
   * and run standalone Quantile Regression + ASSD vs Benchmark test live in browser.
   */
  function analyzeCustomCsv(csvText, benchCdf, gridPct, epsThr) {
    const lines = csvText.trim().split(/\r?\n/).filter((l) => l.trim().length > 0);
    if (lines.length < 110) {
      throw new Error("File CSV cần tối thiểu 110 phiên giao dịch (Date, Close, High, Low, Volume) để huấn luyện mô hình ML.");
    }
    const headers = lines[0].split(/[,\t;]/).map((h) => h.trim().toLowerCase());
    const idxD = headers.findIndex((h) => h.includes("date") || h.includes("ngày"));
    const idxC = headers.findIndex((h) => h.includes("close") || h.includes("đóng") || h.includes("gia"));
    const idxH = headers.findIndex((h) => h.includes("high") || h.includes("cao"));
    const idxL = headers.findIndex((h) => h.includes("low") || h.includes("thấp"));
    const idxV = headers.findIndex((h) => h.includes("vol") || h.includes("kl"));
    const idxF = headers.findIndex((h) => h.includes("fnf") || h.includes("foreign") || h.includes("ngoại"));
    if (idxD < 0 || idxC < 0) {
      throw new Error("Không tìm thấy cột Date và Close trong dòng tiêu đề CSV.");
    }

    const rows = [];
    for (let i = 1; i < lines.length; i++) {
      const cols = lines[i].split(/[,\t;]/).map((c) => c.trim());
      const c = parseFloat(cols[idxC]);
      if (!Number.isFinite(c) || c <= 0) continue;
      rows.push({
        date: cols[idxD],
        close: c,
        high: idxH >= 0 && Number.isFinite(parseFloat(cols[idxH])) ? parseFloat(cols[idxH]) : c,
        low: idxL >= 0 && Number.isFinite(parseFloat(cols[idxL])) ? parseFloat(cols[idxL]) : c,
        vol: idxV >= 0 && Number.isFinite(parseFloat(cols[idxV])) ? parseFloat(cols[idxV]) : 100000,
        fnf: idxF >= 0 && Number.isFinite(parseFloat(cols[idxF])) ? parseFloat(cols[idxF]) : null,
      });
    }
    rows.sort((a, b) => a.date.localeCompare(b.date));
    if (rows.length < 110) throw new Error("Số dòng hợp lệ sau khi lọc < 110 phiên.");

    // Clean >14% unadjusted splits and compute returns & rolling indicators
    let repairs = 0;
    const rets = [0];
    const cleanCloses = [rows[0].close];
    for (let i = 1; i < rows.length; i++) {
      let r = rows[i].close / rows[i - 1].close - 1.0;
      if (Math.abs(r) > 0.14) {
        r = 0.0;
        repairs++;
      } else if (Math.abs(r) > 0.075) {
        r = Math.max(-0.07, Math.min(0.07, r));
        repairs++;
      }
      rets.push(r);
      cleanCloses.push(cleanCloses[i - 1] * (1.0 + r));
    }

    const horizon = 22;
    const X = [];
    const y = [];
    const sig = [];
    let ewVar = 0.0004;
    const volsArr = [];
    const cmfArr = [];
    for (let i = 0; i < rows.length; i++) {
      ewVar = 0.06 * rets[i] * rets[i] + 0.94 * ewVar;
      volsArr.push(Math.max(Math.sqrt(ewVar) * Math.sqrt(horizon), 0.02));
      const hl = rows[i].high - rows[i].low;
      const mfm = hl > 1e-6 ? ((rows[i].close - rows[i].low) - (rows[i].high - rows[i].close)) / hl : 0;
      cmfArr.push(mfm);
    }

    const makeFeat = (t) => {
      const r5 = cleanCloses[t] / cleanCloses[Math.max(0, t - 5)] - 1.0;
      const r22 = cleanCloses[t] / cleanCloses[Math.max(0, t - 22)] - 1.0;
      const cmf20 = cmfArr.slice(Math.max(0, t - 19), t + 1).reduce((a, b) => a + b, 0) / 20.0;
      return [
        Math.tanh(rets[t] * 15),
        Math.tanh(r5 * 8),
        Math.tanh(r22 * 4),
        Math.tanh((volsArr[t] - 0.06) * 12),
        Math.tanh(cmf20 * 2.5),
      ];
    };

    for (let t = 25; t < rows.length - horizon; t++) {
      X.push(makeFeat(t));
      y.push(cleanCloses[t + horizon] / cleanCloses[t] - 1.0);
      sig.push(volsArr[t]);
    }
    const z = y.map((yv, i) => Math.max(-4.5, Math.min(4.5, yv / sig[i])));
    const coefs = fitQuantileRidge(X, z, TAUS_LIVE, 8.0);
    const xNow = makeFeat(rows.length - 1);
    const sigNow = volsArr[rows.length - 1];
    const qNowPct = coefs.map((b) => dotVec(b, xNow) * sigNow * 100.0).sort((a, b) => a - b);
    const cdfCustom = window.SDMath.quantilesToCdf(qNowPct, TAUS_LIVE, gridPct);
    const sd = window.SDMath.computeSdMetrics(cdfCustom, benchCdf, gridPct);
    const cls = window.SDMath.classifySd(sd, epsThr);

    const lastCmf20 = cmfArr.slice(-20).reduce((a, b) => a + b, 0) / 20.0;
    const latestFnf = rows.slice(-20).filter((r) => r.fnf !== null);
    const avgFnf = latestFnf.length ? latestFnf.reduce((a, b) => a + b.fnf, 0) / latestFnf.length : null;

    return {
      nDays: rows.length,
      start: rows[0].date,
      end: rows[rows.length - 1].date,
      repairs,
      lastClose: rows[rows.length - 1].close,
      predMeanPct: sd.mean_a * 100.0,
      q10Pct: qNowPct[1],
      q50Pct: qNowPct[4],
      q90Pct: qNowPct[7],
      cmf20: lastCmf20,
      avgFnf,
      sd,
      verdict: cls,
      cdf: cdfCustom,
    };
  }

  return {
    TAUS_LIVE,
    runLiveExperiment,
    analyzeCustomCsv,
  };
})();
