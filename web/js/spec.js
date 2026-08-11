/** Shared shape of a loaded rule. Derived from each bundle; defaults match training. */

export const DEFAULT_CHANNELS = 16;
export const DEFAULT_HIDDEN = 128;
export const ALPHA_IDX = 3;
export const HIDDEN_FROM = 4;

/** Percept width: [self, mean_n, gated_mean_diff, log1p(deg)] */
export function perceptWidth(c) {
  return 3 * c + 1;
}

export function hiddenCount(c, hiddenFrom = HIDDEN_FROM) {
  return Math.max(0, c - hiddenFrom);
}

export function channelIds(c, alphaIdx = ALPHA_IDX) {
  const ids = [alphaIdx];
  for (let i = HIDDEN_FROM; i < c; i++) ids.push(i);
  return ids;
}
