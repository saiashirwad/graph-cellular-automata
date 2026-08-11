import { HIDDEN_FROM } from "../spec.js";
import { cl, swatch } from "./colors.js";

function nearestCentroid(x, i, c, umap) {
  const cent = umap.centroids, K = umap.k, H = umap.hdim;
  let best = 0, bd = Infinity;
  for (let k = 0; k < K; k++) {
    let d = 0;
    for (let j = 0; j < H; j++) {
      const v = x[i * c + HIDDEN_FROM + j] - cent[k * H + j];
      d += v * v;
    }
    if (d < bd) { bd = d; best = k; }
  }
  return best;
}

function pcaProj(x, i, c, pca) {
  const comp = pca.components, mean = pca.mean;
  let p0 = 0, p1 = 0, p2 = 0;
  for (let j = 0; j < 12; j++) {
    const hv = x[i * c + HIDDEN_FROM + j] - mean[j];
    p0 += comp[j] * hv;
    p1 += comp[12 + j] * hv;
    p2 += comp[24 + j] * hv;
  }
  return [
    (cl(0.5 + p0 * 2) * 255) | 0,
    (cl(0.5 + p1 * 2) * 255) | 0,
    (cl(0.5 + p2 * 2) * 255) | 0,
  ];
}

export function buildViews(ctx) {
  const views = [
    {
      id: "rgb", name: "Pattern", desc: "the colors it grows",
      cap: "The pattern the rule was trained to grow.",
      legParts: () => [
        "colors are the pattern the rule was trained to grow",
        swatch("#000,#fff", "dead → alive"),
      ],
      paint: paintRgb,
    },
    {
      id: "act", name: "Activity", desc: "where the rule fires",
      cap: "Bright = the rule just fired there.",
      legParts: () => [
        swatch("rgb(60,50,140),rgb(255,140,175)", "idle → firing"),
        "where the rule is working",
      ],
      paint: paintAct,
    },
    {
      id: "pca", name: "Hidden state", desc: "all 16 channels, squeezed into color",
      cap: "Hidden state, compressed to color.",
      available: c => !!c.pca,
      legParts: () => ["similar color = similar hidden state"],
      paint: paintPca,
    },
  ];
  if (ctx.umap) {
    views.push({
      id: "umap", name: "Cell types", desc: "cells grouped into discrete types",
      cap: "Same color = same learned cell type.",
      legParts: () => ["same color = same learned cell type"],
      paint: paintUmap,
    });
  }
  return views;
}

export function channelView(id) {
  return {
    id,
    chan: true,
    cap: id === 3
      ? "α — the aliveness channel."
      : `Channel ${id} — the rule invented its meaning during training.`,
    legParts: () => [
      swatch("rgb(90,209,232),#0a0a0b,rgb(255,111,145)", "− → +"),
      id === 3 ? "α — aliveness" : "channel " + id + " — meaning learned in training",
    ],
    paint: (g, ctx, geom) => paintChannel(g, ctx, geom, id),
  };
}

function paintRgb(g, ctx, geom) {
  const { n, c, alphaIdx: A, x, glow } = ctx;
  const { px, py, dispPos, dispD, dpr, rCore, rHalo } = geom;
  if (glow) {
    g.globalCompositeOperation = "lighter";
    for (let i = 0; i < n; i++) {
      const a = x[i * c + A];
      if (a < 0.05) continue;
      const R = (cl(x[i * c]) * 255) | 0, G = (cl(x[i * c + 1]) * 255) | 0, B = (cl(x[i * c + 2]) * 255) | 0;
      const X = px(dispPos[i * 2]), Y = py(dispPos[i * 2 + 1]);
      const grad = g.createRadialGradient(X, Y, 0, X, Y, rHalo);
      grad.addColorStop(0, `rgba(${R},${G},${B},${(0.22 * cl(a)).toFixed(3)})`);
      grad.addColorStop(1, `rgba(${R},${G},${B},0)`);
      g.fillStyle = grad;
      g.beginPath(); g.arc(X, Y, rHalo, 0, 6.2832); g.fill();
    }
  }
  g.globalCompositeOperation = "source-over";
  for (let i = 0; i < n; i++) {
    const a = x[i * c + A];
    if (a < 0.05) continue;
    g.fillStyle = `rgba(${(cl(x[i * c]) * 255) | 0},${(cl(x[i * c + 1]) * 255) | 0},${(cl(x[i * c + 2]) * 255) | 0},${cl(a).toFixed(3)})`;
    g.beginPath();
    g.arc(px(dispPos[i * 2]), py(dispPos[i * 2 + 1]),
      rCore * (0.6 + 0.4 * cl(a)) * (1 - 0.25 * dispD[i]), 0, 6.2832);
    g.fill();
  }
}

function paintAct(g, ctx, geom) {
  const { n, c, alphaIdx: A, x, act } = ctx;
  const { px, py, dispPos, rCore } = geom;
  let sum = 0, cnt = 0;
  for (let i = 0; i < n; i++) if (act[i] > 1e-8) { sum += act[i]; cnt++; }
  const scale = 3 * (cnt ? sum / cnt : 1e-6);
  for (let i = 0; i < n; i++) {
    const v = cl(Math.sqrt(act[i] / scale));
    if (v < 0.02 && x[i * c + A] < 0.05) continue;
    const R = 60 + 195 * v, G = 50 + 90 * v, B = 140 + 35 * v;
    g.fillStyle = `rgba(${R | 0},${G | 0},${B | 0},${cl(0.10 + 0.90 * v).toFixed(3)})`;
    g.beginPath();
    g.arc(px(dispPos[i * 2]), py(dispPos[i * 2 + 1]), rCore * (0.5 + 0.7 * cl(v)), 0, 6.2832);
    g.fill();
  }
}

function paintPca(g, ctx, geom) {
  if (!ctx.pca) return;
  const { n, c, alphaIdx: A, x, glow, pca } = ctx;
  const { px, py, dispPos, dispD, rCore, rHalo } = geom;
  if (glow) {
    g.globalCompositeOperation = "lighter";
    for (let i = 0; i < n; i++) {
      const a = x[i * c + A];
      if (a < 0.05) continue;
      const [R, G, B] = pcaProj(x, i, c, pca);
      const X = px(dispPos[i * 2]), Y = py(dispPos[i * 2 + 1]);
      const grad = g.createRadialGradient(X, Y, 0, X, Y, rHalo);
      grad.addColorStop(0, `rgba(${R},${G},${B},${(0.22 * cl(a)).toFixed(3)})`);
      grad.addColorStop(1, `rgba(${R},${G},${B},0)`);
      g.fillStyle = grad;
      g.beginPath(); g.arc(X, Y, rHalo, 0, 6.2832); g.fill();
    }
  }
  g.globalCompositeOperation = "source-over";
  for (let i = 0; i < n; i++) {
    const a = x[i * c + A];
    if (a < 0.05) continue;
    const [R, G, B] = pcaProj(x, i, c, pca);
    g.fillStyle = `rgba(${R},${G},${B},${cl(a).toFixed(3)})`;
    g.beginPath();
    g.arc(px(dispPos[i * 2]), py(dispPos[i * 2 + 1]),
      rCore * (0.6 + 0.4 * cl(a)) * (1 - 0.25 * dispD[i]), 0, 6.2832);
    g.fill();
    g.lineWidth = 0.75 * geom.dpr;
    g.strokeStyle = "rgba(255,255,255,0.28)";
    g.stroke();
  }
}

function paintUmap(g, ctx, geom) {
  if (!ctx.umap) return;
  const { n, c, alphaIdx: A, x, umap } = ctx;
  const { px, py, dispPos, dispD, rCore, dpr } = geom;
  const cols = umap.colors;
  for (let i = 0; i < n; i++) {
    const a = x[i * c + A];
    if (a < 0.05) continue;
    const best = nearestCentroid(x, i, c, umap);
    const R = (cols[best * 3] * 255) | 0, G = (cols[best * 3 + 1] * 255) | 0, B = (cols[best * 3 + 2] * 255) | 0;
    g.fillStyle = `rgba(${R},${G},${B},${cl(a).toFixed(3)})`;
    g.beginPath();
    g.arc(px(dispPos[i * 2]), py(dispPos[i * 2 + 1]),
      rCore * (0.6 + 0.4 * cl(a)) * (1 - 0.25 * dispD[i]), 0, 6.2832);
    g.fill();
    g.lineWidth = 0.75 * dpr;
    g.strokeStyle = "rgba(255,255,255,0.28)";
    g.stroke();
  }
}

function paintChannel(g, ctx, geom, c0) {
  const { n, c, alphaIdx: A, x } = ctx;
  const { px, py, dispPos, rCore, dpr } = geom;
  let s = 1e-6;
  for (let i = 0; i < n; i++) {
    const v = Math.abs(x[i * c + c0]);
    if (v > s) s = v;
  }
  for (let i = 0; i < n; i++) {
    if (x[i * c + A] < 0.05) continue;
    const X = px(dispPos[i * 2]), Y = py(dispPos[i * 2 + 1]);
    g.fillStyle = "rgba(255,255,255,0.10)";
    g.beginPath(); g.arc(X, Y, 1.5 * dpr, 0, 6.2832); g.fill();
    const v = x[i * c + c0] / s;
    const a = cl(Math.abs(v));
    if (a < 0.03) continue;
    g.fillStyle = v > 0 ? `rgba(255,111,145,${a.toFixed(3)})` : `rgba(90,209,232,${a.toFixed(3)})`;
    g.beginPath();
    g.arc(X, Y, rCore * (0.5 + 0.5 * a), 0, 6.2832);
    g.fill();
  }
}

export function findView(views, id) {
  return views.find(v => v.id === id) || views[0];
}
