import { spaceLayout } from "./space.js";
import { shapeLayout, computeSpectralEmbed } from "./spectral.js";
import { stateLayout } from "./state-pca.js";
import { mapLayout } from "./umap-map.js";

const ALL = [spaceLayout, shapeLayout, stateLayout, mapLayout];

export function availableLayouts(ctx) {
  return ALL.filter(l => !l.available || l.available(ctx));
}

export function findLayout(id, ctx) {
  return availableLayouts(ctx).find(l => l.id === id) || spaceLayout;
}

export function initLayoutBuffers(ctx) {
  const { n, off, src } = ctx;
  ctx.P3 = new Float32Array(n * 3);
  ctx.T3 = new Float32Array(n * 3);
  ctx.dispPos = new Float32Array(n * 2);
  ctx.dispD = new Float32Array(n);
  ctx.spec3 = computeSpectralEmbed(n, off, src);
  ctx.stateV = null;
  spaceLayout.target(ctx.P3, ctx);
}

export function updateLayout(ctx) {
  const layout = findLayout(ctx.layoutId, ctx);
  layout.target(ctx.T3, ctx);
  for (let i = 0; i < ctx.n * 3; i++)
    ctx.P3[i] += (ctx.T3[i] - ctx.P3[i]) * 0.12;

  const flat = layout.isFlat(ctx);
  const cy = Math.cos(ctx.rotY), sy = Math.sin(ctx.rotY);
  const cx = Math.cos(ctx.rotX), sx = Math.sin(ctx.rotX);
  const scale = ctx.dim > 2 ? 0.48 : 0.40;

  for (let i = 0; i < ctx.n; i++) {
    const X = ctx.P3[i * 3], Y = ctx.P3[i * 3 + 1], Z = ctx.P3[i * 3 + 2];
    if (flat) {
      ctx.dispPos[i * 2] = (X + 1) / 2;
      ctx.dispPos[i * 2 + 1] = (Y + 1) / 2;
      ctx.dispD[i] = 0;
    } else {
      const x1 = X * cy + Z * sy, z1 = -X * sy + Z * cy;
      const y1 = Y * cx - z1 * sx, z2 = Y * sx + z1 * cx;
      ctx.dispPos[i * 2]     = 0.5 + scale * x1;
      ctx.dispPos[i * 2 + 1] = 0.5 + scale * y1;
      ctx.dispD[i] = z2;
    }
  }
}

export function spaceIsFlat(ctx) {
  return findLayout(ctx.layoutId, ctx).isFlat(ctx);
}
