/**
 * Front-shell visibility for 3-d layouts.
 * Cleanest tear read: don't draw the far side, so a wound is a real hole.
 */

export function isOrbit3d(ctx) {
  return ctx.dim > 2 && ctx.layoutId !== "map";
}

/** Outward normals ≈ pos − mean(neighbors). Call once per graph load. */
export function computeNormals(n, pos, dim, off, src) {
  const nor = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) {
    let mx = 0, my = 0, mz = 0, cnt = 0;
    for (let e = off[i]; e < off[i + 1]; e++) {
      const j = src[e], p = j * dim;
      mx += pos[p]; my += pos[p + 1];
      mz += dim > 2 ? pos[p + 2] : 0.5;
      cnt++;
    }
    if (!cnt) { nor[i * 3 + 2] = 1; continue; }
    mx /= cnt; my /= cnt; mz /= cnt;
    const p = i * dim;
    let nx = pos[p] - mx, ny = pos[p + 1] - my;
    let nz = (dim > 2 ? pos[p + 2] : 0.5) - mz;
    const len = Math.hypot(nx, ny, nz) || 1;
    nor[i * 3] = nx / len;
    nor[i * 3 + 1] = ny / len;
    nor[i * 3 + 2] = nz / len;
  }
  return nor;
}

/**
 * How much the outward normal faces the camera after the same orbit as layout.
 * >0 ≈ front shell, <0 ≈ back shell.
 */
export function facing(nor, i, rotY, rotX) {
  if (!nor) return 1;
  const nx = nor[i * 3], ny = nor[i * 3 + 1], nz = nor[i * 3 + 2];
  const cy = Math.cos(rotY), sy = Math.sin(rotY);
  const cx = Math.cos(rotX), sx = Math.sin(rotX);
  const x1 = nx * cy + nz * sy, z1 = -nx * sy + nz * cy;
  const z2 = ny * sx + z1 * cx;
  // match dispD sign: layout stores z2; front of cloud tracks +z2 in practice
  return z2;
}

/** True if this node sits on the shell facing the user. */
export function onFrontShell(ctx, i) {
  if (!isOrbit3d(ctx)) return true;
  // small negative slack keeps silhouettes from flickering
  return facing(ctx.normals, i, ctx.rotY, ctx.rotX) > -0.05;
}
