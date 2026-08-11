import { divPx } from "./colors.js";

export function drawMatrix(canvas, M, rows, cols, rowIdx, seps) {
  const tmp = document.createElement("canvas");
  tmp.width = cols; tmp.height = rows;
  const img = tmp.getContext("2d").createImageData(cols, rows);
  let s = 1e-6;
  for (let i = 0; i < M.length; i++) {
    const v = Math.abs(M[i]);
    if (v > s) s = v;
  }
  s *= 0.6;
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++)
      divPx(img.data, (r * cols + c) * 4, M[rowIdx(r, c)] / s);
  tmp.getContext("2d").putImageData(img, 0, 0);
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  const w = (canvas.clientWidth || 224) * dpr;
  const h = Math.round(w * rows / cols);
  canvas.width = w; canvas.height = h;
  canvas.style.height = (h / dpr) + "px";
  const g = canvas.getContext("2d");
  g.imageSmoothingEnabled = false;
  g.drawImage(tmp, 0, 0, w, h);
  g.strokeStyle = "rgba(255,255,255,0.25)";
  g.lineWidth = 1;
  for (const r of seps) {
    const y = Math.round(r / rows * h) + 0.5;
    g.beginPath(); g.moveTo(0, y); g.lineTo(w, y); g.stroke();
  }
}

export function renderWeights(ctx) {
  const { w1, w2, c, h } = ctx;
  const inW = 3 * c + 1;
  // W1 is H x inW; show inputs as rows, hidden as columns (drop degree row from seps of C blocks)
  drawMatrix(document.getElementById("w1"), w1, inW, h, (r, col) => col * inW + r, [c, 2 * c, 3 * c]);
  drawMatrix(document.getElementById("w2"), w2, c, h, (r, col) => r * h + col, [4]);
}

export function createTrace(seedGetter, cGetter) {
  const TRACE_W = 240;
  const hist = [];
  const canvas = document.getElementById("trace");
  const g = canvas.getContext("2d");
  return {
    push(x) {
      const c = cGetter();
      const seed = seedGetter();
      hist.push(x.slice(seed * c, seed * c + c));
      if (hist.length > TRACE_W) hist.shift();
    },
    clear() { hist.length = 0; },
    draw() {
      const c = cGetter();
      if (canvas.width !== TRACE_W) { canvas.width = TRACE_W; canvas.height = c; }
      const img = g.createImageData(TRACE_W, c);
      const n = hist.length;
      const scale = new Float32Array(c).fill(1e-6);
      for (let j = 0; j < n; j++)
        for (let ch = 0; ch < c; ch++) {
          const v = Math.abs(hist[j][ch]);
          if (v > scale[ch]) scale[ch] = v;
        }
      for (let j = 0; j < n; j++) {
        const xx = TRACE_W - n + j;
        for (let ch = 0; ch < c; ch++)
          divPx(img.data, (ch * TRACE_W + xx) * 4, hist[j][ch] / scale[ch]);
      }
      g.putImageData(img, 0, 0);
    },
  };
}

export function createSpark() {
  const hist = [];
  const canvas = document.getElementById("spark");
  const g = canvas.getContext("2d");
  return {
    push(L) {
      hist.push(L);
      if (hist.length > 240) hist.shift();
    },
    clear() { hist.length = 0; },
    draw() {
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      const w = canvas.clientWidth * dpr, h = canvas.clientHeight * dpr;
      if (canvas.width !== w) { canvas.width = w; canvas.height = h; }
      g.clearRect(0, 0, w, h);
      if (hist.length < 2) return;
      const lo = Math.log10(1e-4), hi = Math.log10(1);
      const yOf = v => {
        const u = (Math.log10(Math.max(v, 1e-4)) - lo) / (hi - lo);
        return h - 4 - u * (h - 8);
      };
      g.beginPath();
      for (let i = 0; i < hist.length; i++) {
        const xx = (i / (hist.length - 1)) * (w - 8) + 4;
        i ? g.lineTo(xx, yOf(hist[i])) : g.moveTo(xx, yOf(hist[i]));
      }
      g.strokeStyle = "rgba(250,250,250,0.9)";
      g.lineWidth = 1 * dpr;
      g.stroke();
      g.strokeStyle = "rgba(255,255,255,0.12)";
      g.beginPath();
      g.moveTo(0, h - 0.5 * dpr); g.lineTo(w, h - 0.5 * dpr);
      g.stroke();
    },
  };
}
