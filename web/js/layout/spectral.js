/** 3 lowest-frequency Laplacian modes via smoothing + deflation. */

export function computeSpectralEmbed(n, off, src) {
  const V = [0, 1, 2].map(() => Float32Array.from({ length: n }, () => Math.random() - 0.5));
  const tmp = new Float32Array(n);
  for (let it = 0; it < 220; it++) {
    for (let v = 0; v < 3; v++) {
      const a = V[v];
      let m = 0;
      for (let i = 0; i < n; i++) m += a[i];
      m /= n;
      for (let i = 0; i < n; i++) {
        const d = off[i + 1] - off[i] || 1;
        let s = 0;
        for (let e = off[i]; e < off[i + 1]; e++) s += a[src[e]];
        tmp[i] = 0.5 * (a[i] - m) + 0.5 * (s / d - m);
      }
      a.set(tmp);
      for (let u = 0; u < v; u++) {
        const b = V[u];
        let dot = 0;
        for (let i = 0; i < n; i++) dot += a[i] * b[i];
        for (let i = 0; i < n; i++) a[i] -= dot * b[i];
      }
      let nn = 0;
      for (let i = 0; i < n; i++) nn += a[i] * a[i];
      nn = Math.sqrt(nn) || 1;
      for (let i = 0; i < n; i++) a[i] /= nn;
    }
  }
  const out = new Float32Array(n * 3);
  for (let v = 0; v < 3; v++) {
    let mx = 1e-9;
    for (let i = 0; i < n; i++) mx = Math.max(mx, Math.abs(V[v][i]));
    for (let i = 0; i < n; i++) out[i * 3 + v] = V[v][i] / mx * 0.9;
  }
  return out;
}

export const shapeLayout = {
  id: "shape",
  name: "Shape",
  desc: "from connections alone",
  cap: "Laid out from connections alone — no coordinates given.",
  leg: "drag to orbit · click to tear",
  target(out, ctx) {
    out.set(ctx.spec3);
  },
  isFlat() { return false; },
};
