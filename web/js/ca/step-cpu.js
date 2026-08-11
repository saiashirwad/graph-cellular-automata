/**
 * One Graph-NCA step on the CPU. Matches src/gnca/model.py:
 *   percept = [self, mean(nbr), gated_mean_diff, log1p(deg)]
 *   dx = Linear(3C+1, H) → ReLU → Linear(H, C)
 *   x ← (x + dx * bernoulli) * alive_mask
 *
 * Gate is identity when gateW/gateB are zero (2·sigmoid(0) = 1), so pre-#17/#18
 * checkpoints behave as before after zero-pad export.
 *
 * @param {object} s  sim fields: x, act, n, c, h, alphaIdx, off, src, deg,
 *                    w1, b1, w2, gateW, gateB
 * @param {object} [opts]
 * @param {number} [opts.updateRate=0.5]
 * @param {() => number} [opts.rand=Math.random]  inject for tests
 */
export function stepCpu(s, opts = {}) {
  const {
    n, c, h, alphaIdx: A, off, src, deg,
    w1, b1, w2, gateW, gateB, x, act,
  } = s;
  const updateRate = opts.updateRate ?? 0.5;
  const rand = opts.rand ?? Math.random;
  const inW = 3 * c + 1;

  // alive if self or any neighbor has alpha > 0.1 (pre-update, like training)
  const alive = new Uint8Array(n);
  for (let i = 0; i < n; i++) {
    let ok = x[i * c + A] > 0.1;
    if (!ok) {
      for (let e = off[i]; e < off[i + 1]; e++) {
        if (x[src[e] * c + A] > 0.1) { ok = true; break; }
      }
    }
    alive[i] = ok ? 1 : 0;
  }

  // endpoint gate features a = x @ [Wg1; Wg2] → (n, 2c)
  // gateW layout: row-major (c, 3c) = [Wg1 | Wg2 | Wg3]
  const a = new Float32Array(n * 2 * c);
  for (let i = 0; i < n; i++) {
    const xb = i * c;
    const ab = i * 2 * c;
    for (let r = 0; r < c; r++) {
      let s1 = 0, s2 = 0;
      const g1 = r * 3 * c;
      const g2 = g1 + c;
      for (let k = 0; k < c; k++) {
        const xv = x[xb + k];
        s1 += gateW[g1 + k] * xv;
        s2 += gateW[g2 + k] * xv;
      }
      a[ab + r] = s1;
      a[ab + c + r] = s2;
    }
  }

  const xn = new Float32Array(n * c);
  const z = new Float32Array(inW);
  const hid = new Float32Array(h);

  for (let i = 0; i < n; i++) {
    const d = deg[i] > 0 ? deg[i] : 1;
    const inv = 1 / d;
    const ab = i * 2 * c;
    const xb = i * c;

    // mean neighbor + gated mean_diff
    for (let ch = 0; ch < c; ch++) {
      let sum = 0, sumDiff = 0;
      for (let e = off[i]; e < off[i + 1]; e++) {
        const j = src[e];
        const jb = j * c;
        const jAb = j * 2 * c;
        const diff = x[jb + ch] - x[xb + ch];
        // g_ch = 2 * sigmoid(a_src[ch] + a_dst[c+ch] + |d|·Wg3[ch] + b)
        let logit = a[jAb + ch] + a[ab + c + ch] + gateB[ch];
        const g3 = ch * 3 * c + 2 * c;
        for (let k = 0; k < c; k++) {
          const dd = x[jb + k] - x[xb + k];
          logit += gateW[g3 + k] * (dd < 0 ? -dd : dd);
        }
        const g = 2 / (1 + Math.exp(-logit));
        sum += x[jb + ch];
        sumDiff += g * diff;
      }
      const mn = sum * inv;
      z[ch] = x[xb + ch];
      z[c + ch] = mn;
      z[2 * c + ch] = sumDiff * inv;
    }
    z[3 * c] = Math.log1p(deg[i]);

    // MLP
    for (let k = 0; k < h; k++) {
      let acc = b1[k];
      const base = k * inW;
      for (let p = 0; p < inW; p++) acc += w1[base + p] * z[p];
      hid[k] = acc > 0 ? acc : 0;
    }
    const mask = rand() < updateRate ? 1 : 0;
    for (let ch = 0; ch < c; ch++) {
      let acc = 0;
      const base = ch * h;
      for (let k = 0; k < h; k++) acc += w2[base + k] * hid[k];
      xn[xb + ch] = alive[i] ? x[xb + ch] + acc * mask : 0;
    }
  }

  for (let i = 0; i < n; i++) {
    let ssq = 0;
    const xb = i * c;
    for (let ch = 0; ch < c; ch++) {
      const d = xn[xb + ch] - x[xb + ch];
      ssq += d * d;
    }
    act[i] = 0.85 * act[i] + 0.15 * Math.sqrt(ssq);
  }

  s.x = xn;
  s.t++;
  return s;
}
