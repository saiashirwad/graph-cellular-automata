/** Nearest UMAP palette centroid → offline map coordinates. */

import { HIDDEN_FROM } from "../spec.js";

let mapJit = null;

export function mapEmbed(out, ctx) {
  const umap = ctx.umap;
  if (!umap) return;
  const { n, c, x } = ctx;
  const cent = umap.centroids, c2 = umap.cent2d, K = umap.k, Hd = umap.hdim;

  if (!mapJit || mapJit.length !== n * 2) {
    mapJit = new Float32Array(n * 2);
    for (let i = 0; i < n; i++) {
      const h1 = Math.sin(i * 12.9898) * 43758.5453;
      const h2 = Math.sin(i * 78.233) * 12543.217;
      mapJit[i * 2]     = (h1 - Math.floor(h1) - 0.5) * 0.05;
      mapJit[i * 2 + 1] = (h2 - Math.floor(h2) - 0.5) * 0.05;
    }
  }

  for (let i = 0; i < n; i++) {
    let best = 0, bd = Infinity;
    for (let k = 0; k < K; k++) {
      let d = 0;
      for (let j = 0; j < Hd; j++) {
        const v = x[i * c + HIDDEN_FROM + j] - cent[k * Hd + j];
        d += v * v;
      }
      if (d < bd) { bd = d; best = k; }
    }
    out[i * 3]     = c2[best * 2] + mapJit[i * 2];
    out[i * 3 + 1] = c2[best * 2 + 1] + mapJit[i * 2 + 1];
    out[i * 3 + 2] = 0;
  }
}

export const mapLayout = {
  id: "map",
  name: "Map",
  desc: "cluster by cell type",
  cap: "Cells with the same learned role land in the same place.",
  leg: "faint dots = states seen in the training rollout",
  available(ctx) { return !!ctx.umap; },
  target(out, ctx) { mapEmbed(out, ctx); },
  isFlat() { return true; },
};
