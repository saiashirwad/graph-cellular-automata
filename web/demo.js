// Graph Neural Cellular Automata — pure-browser forward pass.
// Reads weights/topology from bundle.js (+ bundle_ws.js), reimplements the
// CA step in JS. No framework, no server, no build step.
"use strict";

// ---- model state (swappable) ----
let N, C, AIDX, pos, off, src, W1, b1, W2, H, TGT, SEED;
let x;                       // node states Float32Array(N*C)
let t = 0, running = true, stepsPerFrame = 2, ghost = true, brushR = 0.08;

function loadBundle(b) {
  N = b.n; C = b.channels; AIDX = b.alive_alpha_idx; H = 128;
  pos = Float32Array.from(b.pos);
  off = Int32Array.from(b.csr_off);
  src = Int32Array.from(b.csr_src);
  W1 = Float32Array.from(b.w1);
  b1 = Float32Array.from(b.b1);
  W2 = Float32Array.from(b.w2);
  TGT = Float32Array.from(b.target);
  SEED = b.seed;
  x = new Float32Array(N * C);
  seed();
}

// ---- canvas ----
const cv = document.getElementById("c");
const ctx = cv.getContext("2d");
const RES = 600;
function toPx(p) { return p * RES; }

function seed() {
  x.fill(0);
  for (let c = AIDX; c < C; c++) x[SEED * C + c] = 1;
  t = 0;
}

// ---- one CA step ----
function step() {
  const alive = new Uint8Array(N);
  for (let i = 0; i < N; i++) {
    let mx = x[i * C + AIDX];
    if (mx > 0.1) { alive[i] = 1; continue; }
    for (let e = off[i]; e < off[i+1]; e++)
      if (x[src[e] * C + AIDX] > 0.1) { mx = 0.2; break; }
    alive[i] = mx > 0.1 ? 1 : 0;
  }
  const xn = new Float32Array(N * C);
  const z = new Float32Array(3 * C);
  const h = new Float32Array(H);
  for (let i = 0; i < N; i++) {
    const d = off[i+1] - off[i] || 1;
    for (let c = 0; c < C; c++) {
      let s = 0;
      for (let e = off[i]; e < off[i+1]; e++) s += x[src[e] * C + c];
      const mn = s / d;
      z[c] = x[i * C + c];
      z[C + c] = mn;
      z[2*C + c] = mn - x[i * C + c];
    }
    for (let k = 0; k < H; k++) {
      let acc = b1[k];
      const base = k * (3 * C);
      for (let c = 0; c < 3*C; c++) acc += W1[base + c] * z[c];
      h[k] = acc > 0 ? acc : 0;
    }
    for (let c = 0; c < C; c++) {
      const base = c * H;
      let acc = 0;
      for (let k = 0; k < H; k++) acc += W2[base + k] * h[k];
      xn[i * C + c] = x[i * C + c] + acc * (Math.random() < 0.5 ? 1 : 0);
    }
  }
  for (let i = 0; i < N; i++)
    if (!alive[i]) for (let c = 0; c < C; c++) xn[i * C + c] = 0;
  x = xn;
  t++;
}

// ---- rendering ----
function draw() {
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, RES, RES);
  if (ghost) {
    for (let i = 0; i < N; i++) {
      const a = TGT[i*4 + 3];
      if (a < 0.05) continue;
      ctx.fillStyle = `rgba(${(TGT[i*4]*255)|0},${(TGT[i*4+1]*255)|0},${(TGT[i*4+2]*255)|0},0.13)`;
      ctx.fillRect(toPx(pos[i*2]) - 3, RES - toPx(pos[i*2+1]) - 3, 7, 7);
    }
  }
  const cl = v => v < 0 ? 0 : v > 1 ? 1 : v;
  for (let i = 0; i < N; i++) {
    const a = x[i * C + AIDX];
    if (a < 0.05) continue;
    ctx.fillStyle = `rgba(${(cl(x[i*C])*255)|0},${(cl(x[i*C+1])*255)|0},${(cl(x[i*C+2])*255)|0},${cl(a)})`;
    ctx.beginPath();
    ctx.arc(toPx(pos[i*2]), RES - toPx(pos[i*2+1]), 4, 0, 6.2832);
    ctx.fill();
  }
}

function loss() {
  let s = 0;
  for (let i = 0; i < N; i++)
    for (let c = 0; c < 4; c++) { const d = x[i*C+c] - TGT[i*4+c]; s += d*d; }
  return s / (N * 4);
}
function alivePct() {
  let a = 0; for (let i = 0; i < N; i++) if (x[i*C+AIDX] > 0.1) a++;
  return a / N * 100;
}

const stats = document.getElementById("stats");
function loop() {
  if (running) for (let s = 0; s < stepsPerFrame; s++) step();
  draw();
  stats.innerHTML = `t = <b>${t}</b><br>alive = <b>${alivePct().toFixed(0)}%</b><br>loss = <b>${loss().toFixed(4)}</b>`;
  requestAnimationFrame(loop);
}

// ---- interaction ----
function paintDamage(ev, radius) {
  const r = cv.getBoundingClientRect();
  const px = (ev.clientX - r.left) / r.width;
  const py = 1 - (ev.clientY - r.top) / r.height;
  const r2 = radius * radius;
  for (let i = 0; i < N; i++) {
    const dx = pos[i*2] - px, dy = pos[i*2+1] - py;
    if (dx*dx + dy*dy < r2)
      for (let c = 0; c < C; c++) x[i*C + c] = 0;
  }
}
let drawing = false;
cv.addEventListener("pointerdown", e => { drawing = true; paintDamage(e, brushR); });
cv.addEventListener("pointermove", e => { if (drawing) paintDamage(e, brushR); });
window.addEventListener("pointerup", () => drawing = false);

document.getElementById("reset").onclick   = seed;
document.getElementById("pause").onclick   = e => { running = !running; e.target.textContent = running ? "⏸ Pause" : "▶ Play"; };
document.getElementById("killall").onclick = () => x.fill(0);
document.getElementById("ghost").onchange  = e => ghost = e.target.checked;
document.getElementById("spd").oninput     = e => { stepsPerFrame = +e.target.value; document.getElementById("spdV").textContent = stepsPerFrame; };
document.getElementById("br").oninput      = e => { brushR = +e.target.value / 100; document.getElementById("brV").textContent = brushR.toFixed(2); };

// model selector
document.getElementById("model").onchange = e => {
  loadBundle(e.target.value === "ring" ? BUNDLE_WS : BUNDLE);
};

// ---- init ----
loadBundle(BUNDLE);
loop();
