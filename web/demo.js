// Graph Neural Cellular Automata — pure-browser forward pass.
// Reads weights/topology from bundle.js (+ bundle_ws.js), reimplements the
// CA step in JS. No framework, no server, no build step.
"use strict";

// ---- model state (swappable) ----
let N, C, AIDX, pos, off, src, W1, b1, W2, H, TGT, SEED;
let x;                       // node states Float32Array(N*C)
let t = 0, running = true, stepsPerFrame = 2, brushR = 0.08;
let showGhost = true, showEdges = true, glow = true;
let view = "rgb";            // "rgb" | "act" | channel index
let act, actMax = 1e-6;      // per-node EMA of update magnitude

// ---- canvas / sizing ----
const cv = document.getElementById("c");
const ctx = cv.getContext("2d");
let W = 600, DPR = 1;

function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  const s = cv.clientWidth || 600;
  W = Math.round(s * DPR);
  cv.width = cv.height = W;
}
new ResizeObserver(resize).observe(cv);

const px = p => p * W;
const py = p => W - p * W;
function renderEdgeLayer() {}                     // edges are drawn per-frame now

// ---- layouts ----
// P3: displayed 3D coords (lerped), T3: target coords for current layout.
// dispPos: projected screen coords (fractions, y-up); dispD: depth in [-1,1].
let layout = "space";
let P3, T3, spec3, dispPos, dispD;
let rotY = 0.5, rotX = 0.35, autoSpin = true;

function initLayouts() {
  P3 = new Float32Array(N * 3);
  T3 = new Float32Array(N * 3);
  dispPos = new Float32Array(N * 2);
  dispD = new Float32Array(N);
  for (let i = 0; i < N; i++) {                   // start at trained positions
    P3[i*3]   = pos[i*2] * 2 - 1;
    P3[i*3+1] = pos[i*2+1] * 2 - 1;
    P3[i*3+2] = 0;
  }
  spec3 = spectralEmbed();
  stateV = null;
}

// 3 lowest-frequency Laplacian modes via smoothing + deflation:
// pure topology, no trained coordinates involved.
function spectralEmbed() {
  const V = [0, 1, 2].map(() => Float32Array.from({length: N}, () => Math.random() - 0.5));
  const tmp = new Float32Array(N);
  for (let it = 0; it < 220; it++) {
    for (let v = 0; v < 3; v++) {
      const a = V[v];
      let m = 0; for (let i = 0; i < N; i++) m += a[i]; m /= N;
      for (let i = 0; i < N; i++) {               // v <- 0.5 v + 0.5 mean(neighbors)
        const d = off[i+1] - off[i] || 1;
        let s = 0;
        for (let e = off[i]; e < off[i+1]; e++) s += a[src[e]];
        tmp[i] = 0.5 * (a[i] - m) + 0.5 * (s / d - m);
      }
      a.set(tmp);
      for (let u = 0; u < v; u++) {               // orthogonalize
        const b = V[u]; let dot = 0;
        for (let i = 0; i < N; i++) dot += a[i] * b[i];
        for (let i = 0; i < N; i++) a[i] -= dot * b[i];
      }
      let n = 0; for (let i = 0; i < N; i++) n += a[i] * a[i];
      n = Math.sqrt(n) || 1;
      for (let i = 0; i < N; i++) a[i] /= n;
    }
  }
  const out = new Float32Array(N * 3);
  for (let v = 0; v < 3; v++) {
    let mx = 1e-9;
    for (let i = 0; i < N; i++) mx = Math.max(mx, Math.abs(V[v][i]));
    for (let i = 0; i < N; i++) out[i*3 + v] = V[v][i] / mx * 0.9;
  }
  return out;
}

// live 3D PCA of the 12 hidden channels (warm-started power iteration)
let stateV = null;
function stateEmbed() {
  const D = 12, o = 4;
  const mu = new Float32Array(D);
  let na = 0;
  for (let i = 0; i < N; i++) {
    if (x[i*C + AIDX] < 0.05) continue;
    na++;
    for (let d = 0; d < D; d++) mu[d] += x[i*C + o + d];
  }
  if (na > 2) for (let d = 0; d < D; d++) mu[d] /= na;
  const S = new Float32Array(D * D);              // covariance (alive nodes)
  for (let i = 0; i < N; i++) {
    if (x[i*C + AIDX] < 0.05) continue;
    for (let a = 0; a < D; a++) {
      const va = x[i*C + o + a] - mu[a];
      for (let b = a; b < D; b++) S[a*D + b] += va * (x[i*C + o + b] - mu[b]);
    }
  }
  for (let a = 0; a < D; a++) for (let b = 0; b < a; b++) S[a*D + b] = S[b*D + a];
  if (!stateV) stateV = [0,1,2].map(() => Float32Array.from({length: D}, () => Math.random() - 0.5));
  for (let it = 0; it < 4; it++) {
    for (let v = 0; v < 3; v++) {
      const a = stateV[v], t = new Float32Array(D);
      for (let r = 0; r < D; r++) { let s = 0; for (let c2 = 0; c2 < D; c2++) s += S[r*D + c2] * a[c2]; t[r] = s; }
      for (let u = 0; u < v; u++) {
        const b = stateV[u]; let dot = 0;
        for (let d = 0; d < D; d++) dot += t[d] * b[d];
        for (let d = 0; d < D; d++) t[d] -= dot * b[d];
      }
      let n = 0; for (let d = 0; d < D; d++) n += t[d] * t[d];
      n = Math.sqrt(n) || 1;
      for (let d = 0; d < D; d++) a[d] = t[d] / n;
    }
  }
  // project + robust scale
  const proj = new Float32Array(N * 3);
  const sd = [1e-9, 1e-9, 1e-9];
  for (let i = 0; i < N; i++)
    for (let v = 0; v < 3; v++) {
      let s = 0;
      for (let d = 0; d < D; d++) s += stateV[v][d] * (x[i*C + o + d] - mu[d]);
      proj[i*3 + v] = s;
      if (x[i*C + AIDX] >= 0.05) sd[v] += s * s;
    }
  for (let v = 0; v < 3; v++) sd[v] = Math.sqrt(sd[v] / Math.max(na, 1)) * 2.2;
  for (let i = 0; i < N; i++)
    for (let v = 0; v < 3; v++) {
      const s = proj[i*3 + v] / sd[v];
      proj[i*3 + v] = s < -1 ? -1 : s > 1 ? 1 : s;
    }
  return proj;
}

function updateLayout() {
  if (layout === "space") {
    for (let i = 0; i < N; i++) {
      T3[i*3] = pos[i*2] * 2 - 1; T3[i*3+1] = pos[i*2+1] * 2 - 1; T3[i*3+2] = 0;
    }
  } else if (layout === "shape") {
    T3.set(spec3);
  } else {
    T3.set(stateEmbed());
  }
  for (let i = 0; i < N * 3; i++) P3[i] += (T3[i] - P3[i]) * 0.12;
  // project (orthographic with soft depth cue)
  const flat = layout === "space";
  const cy = Math.cos(rotY), sy = Math.sin(rotY);
  const cx = Math.cos(rotX), sx = Math.sin(rotX);
  for (let i = 0; i < N; i++) {
    const X = P3[i*3], Y = P3[i*3+1], Z = P3[i*3+2];
    if (flat) {
      dispPos[i*2] = (X + 1) / 2; dispPos[i*2+1] = (Y + 1) / 2; dispD[i] = 0;
    } else {
      const x1 = X * cy + Z * sy, z1 = -X * sy + Z * cy;
      const y1 = Y * cx - z1 * sx, z2 = Y * sx + z1 * cx;
      dispPos[i*2]   = 0.5 + 0.40 * x1;
      dispPos[i*2+1] = 0.5 + 0.40 * y1;
      dispD[i] = z2;
    }
  }
}

// ---- edge manipulation: keep full edge list, rebuild CSR when changed ----
let fullEdges = null;   // Int32Array of [src0,dst0,src1,dst1,...]
let activeMask = null;  // Uint8Array, 1=active

function storeEdges() {
  const list = [];
  for (let i = 0; i < N; i++)
    for (let e = off[i]; e < off[i+1]; e++) list.push(src[e], i);
  fullEdges = Int32Array.from(list);
  activeMask = new Uint8Array(list.length / 2).fill(1);
}

function rebuildCSR() {
  let count = 0;
  for (let i = 0; i < activeMask.length; i++) if (activeMask[i]) count++;
  const newOff = new Int32Array(N + 1);
  const newSrc = new Int32Array(count);
  let idx = 0;
  for (let i = 0; i < activeMask.length; i++) {
    if (activeMask[i]) {
      newOff[fullEdges[i * 2 + 1] + 1]++;
      newSrc[idx++] = fullEdges[i * 2];
    }
  }
  for (let i = 0; i < N; i++) newOff[i+1] += newOff[i];
  off = newOff;
  src = newSrc;
  renderEdgeLayer();
}

function loadBundle(b) {
  N = b.n; C = b.channels; AIDX = b.alive_alpha_idx; H = 128;
  pos = Float32Array.from(b.pos);
  off = Int32Array.from(b.csr_off);
  src = Int32Array.from(b.csr_src);
  W1 = Float32Array.from(b.w1);
  b1 = Float32Array.from(b.b1);
  W2 = Float32Array.from(b.w2);
  TGT = Float32Array.from(b.target);
  SEED = b.seed;
  x = new Float32Array(N * C);
  act = new Float32Array(N);
  actMax = 1e-6;
  lossHist.length = 0;
  storeEdges();
  initLayouts();
  updateLayout();
  renderWeights();
  seed();
}

function seed() {
  x.fill(0);
  for (let c = AIDX; c < C; c++) x[SEED * C + c] = 1;
  t = 0;
  traceHist.length = 0;
  if (dispPos) ripples.push({ x: dispPos[SEED*2], y: dispPos[SEED*2+1], t0: performance.now(), col: "87,217,160" });
}

// ---- one CA step (numerically identical to the PyTorch model) ----
function step() {
  const alive = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    let mx = x[i * C + AIDX];
    if (mx > 0.1) { alive[i] = 1; continue; }
    for (let e = off[i]; e < off[i+1]; e++)
      if (x[src[e] * C + AIDX] > 0.1) { mx = 0.2; break; }
    alive[i] = mx > 0.1 ? 1 : 0;
  }
  const xn = new Float32Array(N * C);
  const z = new Float32Array(3 * C);
  const h = new Float32Array(H);
  for (let i = 0; i < N; i++) {
    const d = off[i+1] - off[i] || 1;
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let e = off[i]; e < off[i+1]; e++) s += x[src[e] * C + c];
      const mn = s / d;
      z[c] = x[i * C + c];
      z[C + c] = mn;
      z[2*C + c] = mn - x[i * C + c];
    }
    for (let k = 0; k < H; k++) {
      let acc = b1[k];
      const base = k * (3 * C);
      for (let c = 0; c < 3*C; c++) acc += W1[base + c] * z[c];
      h[k] = acc > 0 ? acc : 0;
    }
    for (let c = 0; c < C; c++) {
      const base = c * H;
      let acc = 0;
      for (let k = 0; k < H; k++) acc += W2[base + k] * h[k];
      xn[i * C + c] = x[i * C + c] + acc * (Math.random() < 0.5 ? 1 : 0);
    }
  }
  for (let i = 0; i < N; i++)
    if (!alive[i]) for (let c = 0; c < C; c++) xn[i * C + c] = 0;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (let c = 0; c < C; c++) { const d = xn[i*C+c] - x[i*C+c]; s += d*d; }
    act[i] = 0.85 * act[i] + 0.15 * Math.sqrt(s);
  }
  x = xn;
  t++;
}

// ---- effects ----
const ripples = [];   // {x, y, t0, col}
const RIPPLE_MS = 650;

function drawRipples(now) {
  for (let i = ripples.length - 1; i >= 0; i--) {
    const r = ripples[i];
    const u = (now - r.t0) / RIPPLE_MS;
    if (u >= 1) { ripples.splice(i, 1); continue; }
    ctx.strokeStyle = `rgba(${r.col},${(0.5 * (1 - u)).toFixed(3)})`;
    ctx.lineWidth = 1.5 * DPR;
    ctx.beginPath();
    ctx.arc(px(r.x), py(r.y), (0.015 + 0.12 * u) * W, 0, 6.2832);
    ctx.stroke();
  }
}

// pointer state for the brush cursor / orbit
let ptr = null, drawing = false;
let orbiting = false, moved = false, downPos = null;

// ---- rendering ----
const cl = v => v < 0 ? 0 : v > 1 ? 1 : v;

function draw(now) {
  ctx.clearRect(0, 0, W, W);
  if (showEdges) {
    ctx.lineWidth = Math.max(1, 0.5 * DPR);
    ctx.strokeStyle = "rgba(180,200,190,0.08)";
    ctx.beginPath();
    for (let i = 0; i < N; i++)
      for (let k = off[i]; k < off[i+1]; k++) {
        const j = src[k];
        if (j <= i) continue;
        ctx.moveTo(px(dispPos[i*2]), py(dispPos[i*2+1]));
        ctx.lineTo(px(dispPos[j*2]), py(dispPos[j*2+1]));
      }
    ctx.stroke();
  }

  if (showGhost && view === "rgb" && layout === "space") {
    ctx.globalCompositeOperation = "source-over";
    for (let i = 0; i < N; i++) {
      const a = TGT[i*4 + 3];
      if (a < 0.05) continue;
      ctx.fillStyle = `rgba(${(TGT[i*4]*255)|0},${(TGT[i*4+1]*255)|0},${(TGT[i*4+2]*255)|0},0.14)`;
      ctx.beginPath();
      ctx.arc(px(dispPos[i*2]), py(dispPos[i*2+1]), 2.4 * DPR, 0, 6.2832);
      ctx.fill();
    }
  }

  const rCore = 3.2 * DPR, rHalo = 9 * DPR;
  if (view === "rgb") {
    if (glow) {
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < N; i++) {
        const a = x[i * C + AIDX];
        if (a < 0.05) continue;
        const R = (cl(x[i*C])*255)|0, G = (cl(x[i*C+1])*255)|0, B = (cl(x[i*C+2])*255)|0;
        const g = ctx.createRadialGradient(px(dispPos[i*2]), py(dispPos[i*2+1]), 0,
                                           px(dispPos[i*2]), py(dispPos[i*2+1]), rHalo);
        g.addColorStop(0, `rgba(${R},${G},${B},${(0.22*cl(a)).toFixed(3)})`);
        g.addColorStop(1, `rgba(${R},${G},${B},0)`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(px(dispPos[i*2]), py(dispPos[i*2+1]), rHalo, 0, 6.2832);
        ctx.fill();
      }
    }
    ctx.globalCompositeOperation = "source-over";
    for (let i = 0; i < N; i++) {
      const a = x[i * C + AIDX];
      if (a < 0.05) continue;
      ctx.fillStyle = `rgba(${(cl(x[i*C])*255)|0},${(cl(x[i*C+1])*255)|0},${(cl(x[i*C+2])*255)|0},${cl(a).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(px(dispPos[i*2]), py(dispPos[i*2+1]), rCore * (0.6 + 0.4 * cl(a)) * (1 - 0.25 * dispD[i]), 0, 6.2832);
      ctx.fill();
    }
  } else if (view === "act") {
    // where the rule is firing: per-node update magnitude, fire ramp
    let sum = 0, cnt = 0;
    for (let i = 0; i < N; i++) if (act[i] > 1e-8) { sum += act[i]; cnt++; }
    const scale = 3 * (cnt ? sum / cnt : 1e-6);
    for (let i = 0; i < N; i++) {
      const v = cl(Math.sqrt(act[i] / scale));
      if (v < 0.02 && x[i * C + AIDX] < 0.05) continue;
      const R = 90 + 165 * v, G = 55 + 120 * v, B = 35 + 45 * v;   // ember: dim coal -> hot amber
      ctx.fillStyle = `rgba(${R|0},${G|0},${B|0},${cl(0.10 + 0.90 * v).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(px(dispPos[i*2]), py(dispPos[i*2+1]), rCore * (0.5 + 0.7 * cl(v)), 0, 6.2832);
      ctx.fill();
    }
  } else if (view === "pca" && typeof PCA !== "undefined") {
    // PCA of the 12 hidden channels projected to RGB — morphogen fields
    const comp = Float32Array.from(PCA.components);
    const mean = Float32Array.from(PCA.mean);
    if (glow) {
      ctx.globalCompositeOperation = "lighter";
      for (let i = 0; i < N; i++) {
        const a = x[i * C + AIDX];
        if (a < 0.05) continue;
        let p0 = 0, p1 = 0, p2 = 0;
        for (let j = 0; j < 12; j++) {
          const hv = x[i * C + 4 + j] - mean[j];
          p0 += comp[j]      * hv;
          p1 += comp[12 + j] * hv;
          p2 += comp[24 + j] * hv;
        }
        const R = (cl(0.5 + p0 * 2) * 255) | 0;
        const G = (cl(0.5 + p1 * 2) * 255) | 0;
        const B = (cl(0.5 + p2 * 2) * 255) | 0;
        const g = ctx.createRadialGradient(px(dispPos[i*2]), py(dispPos[i*2+1]), 0,
                                           px(dispPos[i*2]), py(dispPos[i*2+1]), rHalo);
        g.addColorStop(0, `rgba(${R},${G},${B},${(0.22*cl(a)).toFixed(3)})`);
        g.addColorStop(1, `rgba(${R},${G},${B},0)`);
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(px(dispPos[i*2]), py(dispPos[i*2+1]), rHalo, 0, 6.2832);
        ctx.fill();
      }
    }
    ctx.globalCompositeOperation = "source-over";
    for (let i = 0; i < N; i++) {
      const a = x[i * C + AIDX];
      if (a < 0.05) continue;
      let p0 = 0, p1 = 0, p2 = 0;
      for (let j = 0; j < 12; j++) {
        const hv = x[i * C + 4 + j] - mean[j];
        p0 += comp[j]      * hv;
        p1 += comp[12 + j] * hv;
        p2 += comp[24 + j] * hv;
      }
      ctx.fillStyle = `rgba(${(cl(0.5+p0*2)*255)|0},${(cl(0.5+p1*2)*255)|0},${(cl(0.5+p2*2)*255)|0},${cl(a).toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(px(dispPos[i*2]), py(dispPos[i*2+1]), rCore * (0.6 + 0.4 * cl(a)) * (1 - 0.25 * dispD[i]), 0, 6.2832);
      ctx.fill();
      ctx.lineWidth = 0.75 * DPR;
      ctx.strokeStyle = "rgba(221,229,224,0.28)";
      ctx.stroke();
    }
  } else {
    // single state channel, diverging: blue = negative, amber = positive
    const c0 = view;
    let s = 1e-6;
    for (let i = 0; i < N; i++) { const v = Math.abs(x[i*C + c0]); if (v > s) s = v; }
    for (let i = 0; i < N; i++) {
      if (x[i * C + AIDX] < 0.05) continue;
      const X = px(dispPos[i*2]), Y = py(dispPos[i*2+1]);
      ctx.fillStyle = "rgba(221,229,224,0.10)";
      ctx.beginPath(); ctx.arc(X, Y, 1.5 * DPR, 0, 6.2832); ctx.fill();
      const v = x[i*C + c0] / s;
      const a = cl(Math.abs(v));
      if (a < 0.03) continue;
      ctx.fillStyle = v > 0 ? `rgba(255,154,92,${a.toFixed(3)})` : `rgba(87,217,160,${a.toFixed(3)})`;
      ctx.beginPath();
      ctx.arc(X, Y, rCore * (0.5 + 0.5 * a), 0, 6.2832);
      ctx.fill();
    }
  }

  drawRipples(now);

  // hover: show who the nearest node listens to
  let hovered = -1;
  if (ptr && !drawing) {
    let hi = -1, hd = 0.035 * 0.035;
    for (let i = 0; i < N; i++) {
      const dx = pos[i*2] - ptr.x, dy = pos[i*2+1] - ptr.y;
      const d = dx*dx + dy*dy;
      if (d < hd) { hd = d; hi = i; }
    }
    if (hi >= 0) {
      const X = px(dispPos[hi*2]), Y = py(dispPos[hi*2+1]);
      ctx.strokeStyle = "rgba(87,217,160,0.55)";
      ctx.lineWidth = 1 * DPR;
      ctx.beginPath();
      for (let e = off[hi]; e < off[hi+1]; e++) {
        const j = src[e];
        ctx.moveTo(X, Y);
        ctx.lineTo(px(dispPos[j*2]), py(dispPos[j*2+1]));
      }
      ctx.stroke();
      ctx.strokeStyle = "rgba(87,217,160,0.85)";
      for (let e = off[hi]; e < off[hi+1]; e++) {
        const j = src[e];
        ctx.beginPath();
        ctx.arc(px(dispPos[j*2]), py(dispPos[j*2+1]), 3 * DPR, 0, 6.2832);
        ctx.stroke();
      }
      ctx.strokeStyle = "rgba(221,229,224,0.9)";
      ctx.lineWidth = 1.2 * DPR;
      ctx.beginPath();
      ctx.arc(X, Y, 4.5 * DPR, 0, 6.2832);
      ctx.stroke();
      hovered = hi;
    }
  }
  hudEl.textContent = hovered >= 0
    ? "node " + hovered + " · deg " + (off[hovered+1] - off[hovered]) +
      " · α " + x[hovered * C + AIDX].toFixed(2)
    : "";

  // brush cursor
  if (ptr) {
    ctx.strokeStyle = drawing ? "rgba(255,130,70,0.95)" : "rgba(221,229,224,0.4)";
    ctx.lineWidth = 1.2 * DPR;
    ctx.setLineDash([4 * DPR, 4 * DPR]);
    ctx.beginPath();
    ctx.arc(px(ptr.x), py(ptr.y), brushR * W, 0, 6.2832);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = drawing ? "rgba(255,130,70,0.95)" : "rgba(221,229,224,0.6)";
    ctx.fillRect(px(ptr.x) - DPR, py(ptr.y) - DPR, 2 * DPR, 2 * DPR);
  }
}

// ---- stats ----
function loss() {
  let s = 0;
  for (let i = 0; i < N; i++)
    for (let c = 0; c < 4; c++) { const d = x[i*C+c] - TGT[i*4+c]; s += d*d; }
  return s / (N * 4);
}
function alivePct() {
  let a = 0; for (let i = 0; i < N; i++) if (x[i*C+AIDX] > 0.1) a++;
  return a / N * 100;
}

// diverging colormap on console black: negative = phosphor, positive = amber
function divPx(data, o, v) {
  const a = Math.min(Math.abs(v), 1);
  const P = [17, 21, 20];                            // console black
  const H = v > 0 ? [255, 154, 92] : [87, 217, 160]; // amber / phosphor
  data[o]   = P[0] + (H[0] - P[0]) * a;
  data[o+1] = P[1] + (H[1] - P[1]) * a;
  data[o+2] = P[2] + (H[2] - P[2]) * a;
  data[o+3] = 255;
}

// ---- seed-node strip chart ----
const traceHist = [];                 // columns of Float32Array(C)
const TRACE_W = 240;
const trace = document.getElementById("trace");
const trctx = trace.getContext("2d");
function drawTrace() {
  if (trace.width !== TRACE_W) { trace.width = TRACE_W; trace.height = C; }
  const img = trctx.createImageData(TRACE_W, C);
  const n = traceHist.length;
  const scale = new Float32Array(C).fill(1e-6);
  for (let j = 0; j < n; j++)
    for (let c = 0; c < C; c++) {
      const v = Math.abs(traceHist[j][c]);
      if (v > scale[c]) scale[c] = v;
    }
  for (let j = 0; j < n; j++) {
    const xx = TRACE_W - n + j;
    for (let c = 0; c < C; c++)
      divPx(img.data, (c * TRACE_W + xx) * 4, traceHist[j][c] / scale[c]);
  }
  trctx.putImageData(img, 0, 0);
}

// ---- weight heatmaps ----
function drawMatrix(canvas, M, rows, cols, rowIdx, seps) {
  // native-res heatmap, then upscale + hairline row-block separators
  const tmp = document.createElement("canvas");
  tmp.width = cols; tmp.height = rows;
  const img = tmp.getContext("2d").createImageData(cols, rows);
  let s = 1e-6;
  for (let i = 0; i < M.length; i++) { const v = Math.abs(M[i]); if (v > s) s = v; }
  s *= 0.6;                            // saturate the tail a bit
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      divPx(img.data, (r * cols + c) * 4, M[rowIdx(r, c)] / s);
  tmp.getContext("2d").putImageData(img, 0, 0);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = (canvas.clientWidth || 224) * dpr;
  const h = Math.round(w * rows / cols);
  canvas.width = w; canvas.height = h;
  canvas.style.height = (h / dpr) + "px";
  const g = canvas.getContext("2d");
  g.imageSmoothingEnabled = false;
  g.drawImage(tmp, 0, 0, w, h);
  g.strokeStyle = "rgba(221,229,224,0.3)";
  g.lineWidth = 1;
  for (const r of seps) {
    const y = Math.round(r / rows * h) + 0.5;
    g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
  }
}
function renderWeights() {
  // W1: H x 3C, show inputs as rows (3C) and hidden units as columns (H)
  drawMatrix(document.getElementById("w1"), W1, 3 * C, H, (r, c) => c * (3 * C) + r, [C, 2 * C]);
  // W2: C x H, output channels as rows
  drawMatrix(document.getElementById("w2"), W2, C, H, (r, c) => r * H + c, [4]);
}

const lossHist = [];
const spark = document.getElementById("spark");
const sctx = spark.getContext("2d");
function drawSpark() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = spark.clientWidth * dpr, h = spark.clientHeight * dpr;
  if (spark.width !== w) { spark.width = w; spark.height = h; }
  sctx.clearRect(0, 0, w, h);
  if (lossHist.length < 2) return;
  const lo = Math.log10(1e-4), hi = Math.log10(1);
  const yOf = v => {
    const u = (Math.log10(Math.max(v, 1e-4)) - lo) / (hi - lo);
    return h - 4 - u * (h - 8);
  };
  sctx.beginPath();
  for (let i = 0; i < lossHist.length; i++) {
    const xx = (i / (lossHist.length - 1)) * (w - 8) + 4;
    i ? sctx.lineTo(xx, yOf(lossHist[i])) : sctx.moveTo(xx, yOf(lossHist[i]));
  }
  sctx.strokeStyle = "rgba(87,217,160,0.9)";
  sctx.lineWidth = 1 * dpr;
  sctx.stroke();
  sctx.strokeStyle = "rgba(221,229,224,0.15)";
  sctx.beginPath();
  sctx.moveTo(0, h - 0.5 * dpr); sctx.lineTo(w, h - 0.5 * dpr);
  sctx.stroke();
}

const $ = id => document.getElementById(id);
const stT = $("stT"), stA = $("stA"), stL = $("stL"), stE = $("stE");
const hudEl = $("hud");
let frame = 0;

function loop(now) {
  if (running) for (let s = 0; s < stepsPerFrame; s++) step();
  if (layout !== "space" && autoSpin && !orbiting) rotY += 0.0035;
  updateLayout();
  draw(now);
  if ((frame++ & 3) === 0) {                    // stats every 4th frame
    const L = loss(), A = alivePct();
    if (running) {
      lossHist.push(L); if (lossHist.length > 240) lossHist.shift();
      traceHist.push(x.slice(SEED * C, SEED * C + C));
      if (traceHist.length > TRACE_W) traceHist.shift();
      drawTrace();
    }
    if (pendingWound && t >= pendingWound) { pendingWound = 0; woundRandom(); }
    stT.textContent = t;
    stA.textContent = A.toFixed(0) + "%";
    stL.textContent = L.toFixed(4);
    stE.textContent = src.length;
    drawSpark();
  }
  requestAnimationFrame(loop);
}

// ---- interaction ----
function canvasPos(ev) {
  const r = cv.getBoundingClientRect();
  return { x: (ev.clientX - r.left) / r.width, y: 1 - (ev.clientY - r.top) / r.height };
}
function paintDamage(p) {
  const r2 = brushR * brushR;
  for (let i = 0; i < N; i++) {
    const dx = dispPos[i*2] - p.x, dy = dispPos[i*2+1] - p.y;
    if (dx*dx + dy*dy < r2)
      for (let c = 0; c < C; c++) x[i*C + c] = 0;
  }
}
cv.addEventListener("pointerdown", e => {
  ptr = canvasPos(e);
  if (layout === "space") {
    drawing = true; paintDamage(ptr);
    ripples.push({ x: ptr.x, y: ptr.y, t0: performance.now(), col: "255,130,70" });
  } else {
    orbiting = true; moved = false; downPos = { x: e.clientX, y: e.clientY };
  }
});
cv.addEventListener("pointermove", e => {
  const p = canvasPos(e);
  if (orbiting && downPos) {
    const dx = e.clientX - downPos.x, dy = e.clientY - downPos.y;
    if (Math.abs(dx) + Math.abs(dy) > 3) moved = true;
    rotY += dx * 0.008;
    rotX = Math.max(-1.4, Math.min(1.4, rotX + dy * 0.008));
    downPos = { x: e.clientX, y: e.clientY };
  }
  ptr = p;
  if (drawing) paintDamage(p);
});
cv.addEventListener("pointerleave", () => ptr = null);
window.addEventListener("pointerup", e => {
  if (orbiting && !moved && ptr) {              // click without drag = damage
    paintDamage(ptr);
    ripples.push({ x: ptr.x, y: ptr.y, t0: performance.now(), col: "255,130,70" });
  }
  drawing = false; orbiting = false; downPos = null;
});

const pauseBtn = $("pause");
function setRunning(r) {
  running = r;
  pauseBtn.textContent = running ? "Pause" : "Play";
}
$("reset").onclick   = seed;
pauseBtn.onclick     = () => setRunning(!running);
$("killall").onclick = () => x.fill(0);
$("ghost").onchange  = e => showGhost = e.target.checked;
$("edges").onchange  = e => showEdges = e.target.checked;
$("glow").onchange   = e => glow = e.target.checked;

function sliderFill(el) {
  const u = (el.value - el.min) / (el.max - el.min) * 100;
  el.style.setProperty("--fill", u + "%");
}
$("spd").oninput = e => {
  stepsPerFrame = +e.target.value;
  $("spdV").textContent = stepsPerFrame + (stepsPerFrame === 1 ? " step / frame" : " steps / frame");
  sliderFill(e.target);
};
$("br").oninput = e => {
  brushR = +e.target.value / 100;
  $("brV").textContent = brushR.toFixed(2);
  sliderFill(e.target);
};
sliderFill($("spd")); sliderFill($("br"));

// view picker
const VIEWS = [
  { id: "rgb", name: "Pattern", desc: "what the loss sees",
    cap: "The pattern the loss trains.",
    leg: "drag to damage \u00b7 hover a node to see its neighbors" },
  { id: "act", name: "Activity", desc: "where the rule fires",
    cap: "Dark ember = just updated.",
    leg: "dark ember = just updated" },
  { id: "pca", name: "Morphogens", desc: "hidden state \u2192 color",
    cap: "Hidden state, compressed to color.",
    leg: "similar color = similar hidden state" },
];
const CHAN_IDS = [3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
CHAN_IDS.forEach(c => VIEWS.push({
  id: c, chan: true,
  cap: c === 3
    ? "\u03b1 \u2014 the aliveness channel."
    : `Hidden channel ${c} \u2014 self-invented.`,
  leg: "amber = positive \u00b7 blue = negative",
}));

const viewMain = $("viewMain"), viewChans = $("viewChans"),
      viewCap = $("viewCap"), legendEl = $("legend");
VIEWS.forEach(v => {
  const b = document.createElement("button");
  if (v.chan) {
    b.className = "chip";
    b.textContent = v.id === 3 ? "\u03b1" : String(v.id);
    viewChans.appendChild(b);
  } else {
    b.className = "vrow";
    const n = document.createElement("span"); n.className = "vname"; n.textContent = v.name;
    const d = document.createElement("span"); d.className = "vdesc"; d.textContent = v.desc;
    b.append(n, d);
    viewMain.appendChild(b);
  }
  b.onclick = () => setView(v.id);
  v.btn = b;
});
function setView(id) {
  view = id;
  const v = VIEWS.find(v => v.id === id);
  VIEWS.forEach(w => w.btn.classList.toggle("active", w === v));
  viewCap.textContent = v.cap;
  updateLegend();
}

// layout picker
const LAYOUTS = [
  { id: "space", name: "Space", desc: "trained 2-d positions",
    cap: "The trained 2-d coordinates.",
    leg: "drag to damage" },
  { id: "shape", name: "Shape", desc: "topology only, 3-d",
    cap: "The wiring\u2019s own shape.",
    leg: "drag to orbit \u00b7 click to damage" },
  { id: "state", name: "State", desc: "cluster by hidden state",
    cap: "Clustered by what they think.",
    leg: "drag to orbit \u00b7 click to damage" },
];
const layoutMain = $("layoutMain"), layoutCap = $("layoutCap");
LAYOUTS.forEach(l => {
  const b = document.createElement("button");
  b.className = "vrow";
  const n = document.createElement("span"); n.className = "vname"; n.textContent = l.name;
  const d = document.createElement("span"); d.className = "vdesc"; d.textContent = l.desc;
  b.append(n, d);
  b.onclick = () => setLayout(l.id);
  l.btn = b;
  layoutMain.appendChild(b);
});
function setLayout(id) {
  layout = id;
  const l = LAYOUTS.find(l => l.id === id);
  LAYOUTS.forEach(m => m.btn.classList.toggle("active", m === l));
  layoutCap.textContent = l.cap;
  updateLegend();
}
function updateLegend() {
  const v = VIEWS.find(v => v.id === view);
  const l = LAYOUTS.find(l => l.id === layout);
  legendEl.textContent = l.leg + " \u00b7 " + (v.leg.startsWith("drag") ? "hover a node to see its neighbors" : v.leg);
}
setView("rgb");
setLayout("space");

// ---- experiments: guided scenarios with a narrator ----
const storyEl = $("story");
let pendingWound = 0;          // CA step at which to auto-wound (0 = none)

function setStory(html) { storyEl.innerHTML = html; }

function woundRandom() {
  // tear a hole centered on a random living node
  const alive = [];
  for (let i = 0; i < N; i++) if (x[i*C + AIDX] > 0.1) alive.push(i);
  if (!alive.length) return;
  const ci = alive[(Math.random() * alive.length) | 0];
  const cx0 = dispPos[ci*2], cy0 = dispPos[ci*2+1];
  for (let i = 0; i < N; i++) {
    const dx = dispPos[i*2] - cx0, dy = dispPos[i*2+1] - cy0;
    if (dx*dx + dy*dy < 0.12 * 0.12)
      for (let c = 0; c < C; c++) x[i*C + c] = 0;
  }
  ripples.push({ x: cx0, y: cy0, t0: performance.now(), col: "255,130,70" });
}

const EXPERIMENTS = [
  { num: "01", name: "Birth", desc: "one cell becomes a pattern",
    story: "One seed, one shared rule. No blueprint \u2014 it builds itself.",
    run() { setLayout("space"); setView("rgb"); restoreEdges(); seed(); setRunning(true); } },
  { num: "02", name: "Injury", desc: "tear a hole, watch it heal",
    story: "A wound. The dark embers are cells rebuilding \u2014 tear your own.",
    run() {
      setLayout("space"); setView("act"); setRunning(true);
      if (alivePct() < 10) { seed(); pendingWound = t + 260; }
      else woundRandom();
    } },
  { num: "03", name: "Split brain", desc: "cut the graph in half",
    story: "The halves can\u2019t talk anymore. Each dreams alone.",
    run() { setLayout("space"); setView("rgb"); restoreEdges(); cutSpatial(0.5); setRunning(true); } },
  { num: "04", name: "X-ray", desc: "see its hidden chemistry",
    story: "Same graph, arranged by what each cell is thinking.",
    run() { restoreEdges(); setLayout("state"); setView("pca"); setRunning(true); } },
  { num: "05", name: "Apocalypse", desc: "kill every cell at once",
    story: "All dead, and it stays dead. Life needs a living neighbor.",
    run() { setLayout("space"); setView("rgb"); x.fill(0); setRunning(true); } },
];

const expsEl = $("exps");
EXPERIMENTS.forEach((ex, k) => {
  const b = document.createElement("button");
  b.className = "exp";
  b.title = ex.desc;
  const n = document.createElement("span"); n.className = "enum"; n.textContent = ex.num;
  const t1 = document.createElement("span"); t1.className = "ename"; t1.textContent = ex.name;
  b.append(n, t1);
  b.onclick = () => runExperiment(k);
  ex.btn = b;
  expsEl.appendChild(b);
});
function runExperiment(k) {
  const ex = EXPERIMENTS[k];
  EXPERIMENTS.forEach(e => e.btn.classList.toggle("active", e === ex));
  ex.run();
  setStory(ex.story);
}
setStory("A living graph: 1024 cells, one learned rule. Poke it.");

// re-render charts when an analysis section is opened (canvases have no
// layout size while the <details> is closed)
document.querySelectorAll("details").forEach(d =>
  d.addEventListener("toggle", () => { renderWeights(); drawTrace(); }));

// model picker
document.querySelectorAll("#models button").forEach(btn => {
  btn.onclick = () => {
    document.querySelectorAll("#models button").forEach(b => b.classList.remove("active"));
    btn.classList.add("active");
    loadBundle(btn.dataset.model === "ring" ? BUNDLE_WS : BUNDLE);
  };
});

// edge cutting
function cutSpatial(yLine) {
  for (let i = 0; i < activeMask.length; i++) {
    const s = fullEdges[i*2], d = fullEdges[i*2+1];
    if ((pos[s*2+1] < yLine) !== (pos[d*2+1] < yLine)) activeMask[i] = 0;
  }
  rebuildCSR();
}
function cutRandom(frac) {
  for (let i = 0; i < activeMask.length; i++)
    if (Math.random() < frac) activeMask[i] = 0;
  rebuildCSR();
}
function restoreEdges() {
  activeMask.fill(1);
  rebuildCSR();
}

// wire up edge-cut buttons
const cutBtn = (id, fn) => { const b = $(id); if (b) b.onclick = fn; };
cutBtn("cutSpatial",   () => cutSpatial(0.5));
cutBtn("cutRandom",    () => cutRandom(0.2));
cutBtn("restoreEdges", () => restoreEdges());

// keyboard shortcuts
window.addEventListener("keydown", e => {
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); setRunning(!running); }
  else if (e.key === "r") seed();
  else if (e.key === "k") x.fill(0);
  else if (e.key === "g") { const g = $("ghost"); g.checked = !g.checked; showGhost = g.checked; }
  else if (e.key === "v") {
    const i = VIEWS.findIndex(v => v.id === view);
    setView(VIEWS[(i + 1) % VIEWS.length].id);
  }
  else if (e.key === "l") {
    const i = LAYOUTS.findIndex(l => l.id === layout);
    setLayout(LAYOUTS[(i + 1) % LAYOUTS.length].id);
  }
  else if (e.key >= "1" && e.key <= "5") runExperiment(+e.key - 1);
  else if (e.key === "e") cutRandom(0.2);
  else if (e.key === "E") restoreEdges();
});

// ---- init ----
resize();
loadBundle(BUNDLE);
requestAnimationFrame(loop);
