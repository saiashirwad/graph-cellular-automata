/** Live 3D PCA of the hidden channels (warm-started power iteration). */

import { HIDDEN_FROM } from "../spec.js";

export function stateEmbed(ctx) {
  const { n, c, alphaIdx: A, x } = ctx;
  const D = c - HIDDEN_FROM;
  const o = HIDDEN_FROM;
  const mu = new Float32Array(D);
  let na = 0;
  for (let i = 0; i < n; i++) {
    if (x[i * c + A] < 0.05) continue;
    na++;
    for (let d = 0; d < D; d++) mu[d] += x[i * c + o + d];
  }
  if (na > 2) for (let d = 0; d < D; d++) mu[d] /= na;

  const S = new Float32Array(D * D);
  for (let i = 0; i < n; i++) {
    if (x[i * c + A] < 0.05) continue;
    for (let a = 0; a < D; a++) {
      const va = x[i * c + o + a] - mu[a];
      for (let b = a; b < D; b++) S[a * D + b] += va * (x[i * c + o + b] - mu[b]);
    }
  }
  for (let a = 0; a < D; a++)
    for (let b = 0; b < a; b++) S[a * D + b] = S[b * D + a];

  if (!ctx.stateV || ctx.stateV[0].length !== D)
    ctx.stateV = [0, 1, 2].map(() => Float32Array.from({ length: D }, () => Math.random() - 0.5));

  for (let it = 0; it < 4; it++) {
    for (let v = 0; v < 3; v++) {
      const a = ctx.stateV[v], t = new Float32Array(D);
      for (let r = 0; r < D; r++) {
        let s = 0;
        for (let c2 = 0; c2 < D; c2++) s += S[r * D + c2] * a[c2];
        t[r] = s;
      }
      for (let u = 0; u < v; u++) {
        const b = ctx.stateV[u];
        let dot = 0;
        for (let d = 0; d < D; d++) dot += t[d] * b[d];
        for (let d = 0; d < D; d++) t[d] -= dot * b[d];
      }
      let nn = 0;
      for (let d = 0; d < D; d++) nn += t[d] * t[d];
      nn = Math.sqrt(nn) || 1;
      for (let d = 0; d < D; d++) a[d] = t[d] / nn;
    }
  }

  const proj = new Float32Array(n * 3);
  const sd = [1e-9, 1e-9, 1e-9];
  for (let i = 0; i < n; i++)
    for (let v = 0; v < 3; v++) {
      let s = 0;
      for (let d = 0; d < D; d++) s += ctx.stateV[v][d] * (x[i * c + o + d] - mu[d]);
      proj[i * 3 + v] = s;
      if (x[i * c + A] >= 0.05) sd[v] += s * s;
    }
  for (let v = 0; v < 3; v++) sd[v] = Math.sqrt(sd[v] / Math.max(na, 1)) * 2.2;
  for (let i = 0; i < n; i++)
    for (let v = 0; v < 3; v++) {
      const s = proj[i * 3 + v] / sd[v];
      proj[i * 3 + v] = s < -1 ? -1 : s > 1 ? 1 : s;
    }
  return proj;
}

export const stateLayout = {
  id: "state",
  name: "State",
  desc: "cluster by hidden state",
  cap: "Cells with similar hidden states sit together.",
  leg: "drag to orbit · click to tear",
  target(out, ctx) {
    out.set(stateEmbed(ctx));
  },
  isFlat() { return false; },
};
