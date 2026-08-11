/**
 * Load a bundle artifact into the app context.
 * Bundles are ESM default exports from web/artifacts/.
 */
import {
  ALPHA_IDX, DEFAULT_HIDDEN, HIDDEN_FROM, perceptWidth,
} from "./spec.js";
import {
  degreesFromCSR, edgesFromCSR, rebuildCSR, cutRandom, restoreEdges,
} from "./graph/csr.js";
import { initLayoutBuffers, updateLayout } from "./layout/index.js";
import { buildViews, channelView } from "./render/views.js";
import { channelIds } from "./spec.js";

/** Normalize a raw export into typed arrays the engines consume. */
export function normalizeBundle(b) {
  const n = b.n;
  const c = b.channels;
  const h = b.hidden || DEFAULT_HIDDEN;
  const alphaIdx = b.alive_alpha_idx ?? ALPHA_IDX;
  const pos = Float32Array.from(b.pos);
  const dim = b.dim || Math.round(pos.length / n) || 2;
  const off = Int32Array.from(b.csr_off);
  const src = Int32Array.from(b.csr_src);
  const deg = b.deg
    ? Float32Array.from(b.deg)
    : degreesFromCSR(n, off);

  const inW = perceptWidth(c);
  let w1 = Float32Array.from(b.w1);
  // pad pre-degree checkpoints: H x 3C → H x (3C+1)
  if (w1.length === h * 3 * c) {
    const padded = new Float32Array(h * inW);
    for (let k = 0; k < h; k++) {
      padded.set(w1.subarray(k * 3 * c, (k + 1) * 3 * c), k * inW);
    }
    w1 = padded;
  } else if (w1.length !== h * inW) {
    throw new Error(`w1 length ${w1.length} ≠ H·(3C+1)=${h * inW}`);
  }

  const b1 = Float32Array.from(b.b1);
  const w2 = Float32Array.from(b.w2);
  const gateW = b.gate_w
    ? Float32Array.from(b.gate_w)
    : new Float32Array(c * 3 * c); // zeros → identity gate
  const gateB = b.gate_b
    ? Float32Array.from(b.gate_b)
    : new Float32Array(c);

  return {
    n, c, h, dim, alphaIdx,
    pos, off, src, deg,
    w1, b1, w2, gateW, gateB,
    tgt: Float32Array.from(b.target),
    seed: b.seed,
    targetName: b.target_name || "",
    hiddenFrom: HIDDEN_FROM,
    hiddenCount: c - HIDDEN_FROM,
  };
}

export function applyBundle(app, bundle, extras = {}) {
  const m = normalizeBundle(bundle);
  Object.assign(app, m);
  app.pca = extras.pca
    ? {
        components: Float32Array.from(extras.pca.components),
        mean: Float32Array.from(extras.pca.mean),
      }
    : null;
  app.umap = extras.umap
    ? {
        k: extras.umap.k,
        hdim: extras.umap.hdim,
        centroids: Float32Array.from(extras.umap.centroids),
        colors: Float32Array.from(extras.umap.colors),
        cent2d: extras.umap.cent2d ? Float32Array.from(extras.umap.cent2d) : null,
        pts: extras.umap.pts ? Float32Array.from(extras.umap.pts) : null,
      }
    : null;

  const edges = edgesFromCSR(app.n, app.off, app.src);
  app.fullEdges = edges.fullEdges;
  app.activeMask = edges.activeMask;

  app.x = new Float32Array(app.n * app.c);
  app.act = new Float32Array(app.n);
  app.t = 0;
  app.dying = [];

  // views depend on optional artifacts
  const named = buildViews(app);
  app.chanIds = channelIds(app.c, app.alphaIdx);
  app.chanViews = app.chanIds.map(channelView);
  app.views = [...named, ...app.chanViews];

  initLayoutBuffers(app);
  updateLayout(app);
  seed(app);
  return app;
}

export function seed(app) {
  app.x.fill(0);
  for (let ch = app.alphaIdx; ch < app.c; ch++)
    app.x[app.seed * app.c + ch] = 1;
  app.act.fill(0);
  app.t = 0;
  app.dying = [];
  app.engine?.markStateDirty();
}

/** Snapshot a living node into the soft dissolve list, then zero it. */
function killNode(app, i, now) {
  const { n, c, alphaIdx: A, x, dispPos } = app;
  const a = x[i * c + A];
  if (a < 0.08) {
    for (let ch = 0; ch < c; ch++) x[i * c + ch] = 0;
    return false;
  }
  if (!app.dying) app.dying = [];
  // cap list so a huge brush stroke stays smooth
  if (app.dying.length < 800) {
    const cl = v => (v < 0 ? 0 : v > 1 ? 1 : v);
    const phase = Math.sin(i * 12.9898) * 43758.5453;
    const h = phase - Math.floor(phase);
    app.dying.push({
      x: dispPos[i * 2],
      y: dispPos[i * 2 + 1],
      dx: (h - 0.5) * 0.03,
      dy: (Math.sin(i * 78.233) * 0.5 + 0.5 - 0.5) * 0.02,
      R: (cl(x[i * c]) * 255) | 0,
      G: (cl(x[i * c + 1]) * 255) | 0,
      B: (cl(x[i * c + 2]) * 255) | 0,
      a: cl(a),
      r: 3.2 * (0.6 + 0.4 * cl(a)), // css-ish px; draw scales with dpr via arc
      t0: now,
      phase: h * 6.28,
    });
  }
  for (let ch = 0; ch < c; ch++) x[i * c + ch] = 0;
  return true;
}

export function clearAll(app) {
  const now = performance.now();
  // dissolve a sampling of live nodes so kill-all still feels soft
  let live = 0;
  for (let i = 0; i < app.n; i++) if (app.x[i * app.c + app.alphaIdx] > 0.1) live++;
  const stride = Math.max(1, (live / 120) | 0);
  let k = 0;
  for (let i = 0; i < app.n; i++) {
    if (app.x[i * app.c + app.alphaIdx] < 0.1) continue;
    if ((k++ % stride) === 0) killNode(app, i, now);
    else for (let ch = 0; ch < app.c; ch++) app.x[i * app.c + ch] = 0;
  }
  app.act.fill(0);
  app.engine?.markStateDirty();
}

export function paintDamage(app, p) {
  const r2 = app.brushR * app.brushR;
  const now = performance.now();
  let killed = 0;
  for (let i = 0; i < app.n; i++) {
    const dx = app.dispPos[i * 2] - p.x, dy = app.dispPos[i * 2 + 1] - p.y;
    if (dx * dx + dy * dy >= r2) continue;
    if (killNode(app, i, now)) killed++;
  }
  if (killed) app.engine?.markStateDirty();
  return killed;
}

/** Soft breath at a display point — used instead of explosive ripples. */
export function softSigh(app, x, y, col = "210,170,145") {
  if (!app.ripples) app.ripples = [];
  app.ripples.push({
    x, y, t0: performance.now(), col,
    ms: 1100, r0: 0.008, r1: 0.055, alpha: 0.22, width: 1.0,
  });
}

export function applyEdgeCut(app, frac) {
  cutRandom(app.activeMask, frac);
  const { off, src, deg } = rebuildCSR(app.n, app.fullEdges, app.activeMask);
  app.off = off; app.src = src; app.deg = deg;
  app.engine?.markGraphDirty(off, src, deg);
}

export function applyEdgeRestore(app) {
  restoreEdges(app.activeMask);
  const { off, src, deg } = rebuildCSR(app.n, app.fullEdges, app.activeMask);
  app.off = off; app.src = src; app.deg = deg;
  app.engine?.markGraphDirty(off, src, deg);
}
