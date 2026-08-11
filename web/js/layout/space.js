/** Trained embedding coordinates → display cube [-1, 1]. */

export const spaceLayout = {
  id: "space",
  name: "Space",
  desc: "the real shape",
  cap: "Each node at its real position in the shape.",
  leg: "drag to tear",
  target(out, ctx) {
    const { n, pos, dim } = ctx;
    for (let i = 0; i < n; i++) {
      const o = i * 3;
      const p = i * dim;
      out[o]     = pos[p] * 2 - 1;
      out[o + 1] = pos[p + 1] * 2 - 1;
      out[o + 2] = dim > 2 ? pos[p + 2] * 2 - 1 : 0;
    }
  },
  isFlat(ctx) {
    return ctx.dim < 3;
  },
};
