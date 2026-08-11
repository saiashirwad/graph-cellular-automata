/** Edge list bookkeeping: full undirected edge pairs + active mask → CSR. */

export function edgesFromCSR(n, off, src) {
  const list = [];
  for (let i = 0; i < n; i++)
    for (let e = off[i]; e < off[i + 1]; e++) list.push(src[e], i);
  return {
    fullEdges: Int32Array.from(list),
    activeMask: new Uint8Array(list.length / 2).fill(1),
  };
}

export function degreesFromCSR(n, off) {
  const deg = new Float32Array(n);
  for (let i = 0; i < n; i++) deg[i] = off[i + 1] - off[i];
  return deg;
}

export function rebuildCSR(n, fullEdges, activeMask) {
  let count = 0;
  for (let i = 0; i < activeMask.length; i++) if (activeMask[i]) count++;
  const off = new Int32Array(n + 1);
  const src = new Int32Array(count);
  let idx = 0;
  for (let i = 0; i < activeMask.length; i++) {
    if (!activeMask[i]) continue;
    off[fullEdges[i * 2 + 1] + 1]++;
    src[idx++] = fullEdges[i * 2];
  }
  for (let i = 0; i < n; i++) off[i + 1] += off[i];
  return { off, src, deg: degreesFromCSR(n, off) };
}

export function cutRandom(activeMask, frac) {
  for (let i = 0; i < activeMask.length; i++)
    if (Math.random() < frac) activeMask[i] = 0;
}

export function restoreEdges(activeMask) {
  activeMask.fill(1);
}
