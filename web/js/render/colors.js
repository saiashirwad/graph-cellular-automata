export const cl = v => (v < 0 ? 0 : v > 1 ? 1 : v);

/** Diverging colormap on midnight: negative = cyan, positive = rose. */
export function divPx(data, o, v) {
  const a = Math.min(Math.abs(v), 1);
  const P = [10, 10, 11];
  const H = v > 0 ? [255, 111, 145] : [90, 209, 232];
  data[o]     = P[0] + (H[0] - P[0]) * a;
  data[o + 1] = P[1] + (H[1] - P[1]) * a;
  data[o + 2] = P[2] + (H[2] - P[2]) * a;
  data[o + 3] = 255;
}

export function swatch(stops, lab) {
  return `<i class="sw"><b style="background:linear-gradient(90deg,${stops})"></b>${lab}</i>`;
}
