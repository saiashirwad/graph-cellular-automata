import { availableLayouts, findLayout, spaceIsFlat } from "../layout/index.js";
import { findView } from "../render/views.js";
import { updateLegend } from "../render/draw.js";
import { renderWeights } from "../render/charts.js";
import { seed, clearAll, paintDamage, applyEdgeCut, applyEdgeRestore } from "../model.js";

export const SPEEDS = [1 / 60, 1 / 20, 1 / 8, 1 / 4, 1 / 2, 1, 2, 4, 8];

const $ = id => document.getElementById(id);

function sliderFill(el) {
  const u = (el.value - el.min) / (el.max - el.min) * 100;
  el.style.setProperty("--fill", u + "%");
}

export function wireControls(app, hooks) {
  const pauseBtn = $("pause");

  function setRunning(r) {
    app.running = r;
    pauseBtn.textContent = r ? "Pause" : "Play";
  }
  app.setRunning = setRunning;

  function setSpeedIdx(i) {
    const el = $("spd");
    el.value = i;
    app.stepsPerFrame = SPEEDS[i];
    const sps = Math.round(app.stepsPerFrame * 60);
    $("spdV").textContent = "~" + sps + (sps === 1 ? " step/s" : " steps/s");
    sliderFill(el);
  }
  app.setSpeedIdx = setSpeedIdx;

  function glideSpeedTo(target) {
    clearTimeout(app.spdGlide);
    const tick = () => {
      const cur = +$("spd").value;
      if (cur <= target) { app.spdGlide = 0; return; }
      setSpeedIdx(cur - 1);
      app.spdGlide = setTimeout(tick, 90);
    };
    tick();
  }
  app.glideSpeedTo = glideSpeedTo;

  function brushLabel() {
    if (!app.dispPos) return;
    let n = 0;
    if (app.ptr) {
      const r2 = app.brushR * app.brushR;
      for (let i = 0; i < app.n; i++) {
        const dx = app.dispPos[i * 2] - app.ptr.x, dy = app.dispPos[i * 2 + 1] - app.ptr.y;
        if (dx * dx + dy * dy < r2) n++;
      }
    } else {
      n = Math.round(app.n * Math.PI * app.brushR * app.brushR);
    }
    $("brV").textContent = "~" + n + (n === 1 ? " node" : " nodes");
  }
  app.brushLabel = brushLabel;

  $("reset").onclick = () => hooks.onSeed();
  pauseBtn.onclick = () => setRunning(!app.running);
  $("killall").onclick = () => { clearAll(app); };
  $("edges").onchange = e => { app.showEdges = e.target.checked; };
  $("spin").onchange = e => { app.autoSpin = e.target.checked; };

  $("spd").oninput = e => {
    app.autoFf = false;
    clearTimeout(app.spdGlide);
    setSpeedIdx(+e.target.value);
  };
  $("br").oninput = e => {
    app.brushR = +e.target.value / 100;
    brushLabel();
    sliderFill(e.target);
  };
  sliderFill($("spd"));
  sliderFill($("br"));

  // layout picker
  const layoutMain = $("layoutMain"), layoutCap = $("layoutCap");
  const layouts = availableLayouts(app);
  app.layoutButtons = [];
  layouts.forEach(l => {
    const b = document.createElement("button");
    b.className = "vrow";
    b.title = l.desc;
    const n = document.createElement("span"); n.className = "vname"; n.textContent = l.name;
    const d = document.createElement("span"); d.className = "vdesc"; d.textContent = l.desc;
    b.append(n, d);
    b.onclick = () => setLayout(l.id);
    l.btn = b;
    app.layoutButtons.push(l);
    layoutMain.appendChild(b);
  });

  function setLayout(id) {
    app.layoutId = id;
    const l = findLayout(id, app);
    app.layoutButtons.forEach(m => m.btn.classList.toggle("active", m.id === id));
    layoutCap.textContent = l.cap;
    // refresh space caption for dim
    const space = app.layoutButtons.find(x => x.id === "space");
    if (space) {
      space.desc = app.dim > 2 ? "the real shape, 3-d" : "the real shape, 2-d";
      space.cap = "Each node at its real position in the shape.";
      space.leg = app.dim > 2 ? "drag to orbit · click to tear" : "drag to tear";
      if (space.btn) {
        space.btn.title = space.desc;
        space.btn.querySelector(".vdesc").textContent = space.desc;
      }
    }
    updateLegend(app);
  }
  app.setLayout = setLayout;

  // view picker + channel filmstrip
  const viewMain = $("viewMain"), chanStrip = $("chanstrip");
  const TH = 54;
  app.thumbs = [];

  function makeThumb(ch) {
    const canvas = document.createElement("canvas");
    canvas.className = "cth";
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const s = Math.round(TH * dpr);
    canvas.width = s; canvas.height = s;
    app.thumbs.push({ g: canvas.getContext("2d"), s, c: ch });
    return canvas;
  }

  app.views.filter(v => !v.chan).forEach(v => {
    const b = document.createElement("button");
    b.className = "vrow";
    b.title = v.desc;
    const n = document.createElement("span"); n.className = "vname"; n.textContent = v.name;
    const d = document.createElement("span"); d.className = "vdesc"; d.textContent = v.desc;
    b.append(n, d);
    b.onclick = () => setView(v.id);
    v.btn = b;
    viewMain.appendChild(b);
  });

  app.chanIds.forEach(ch => {
    const v = findView(app.views, ch);
    const b = document.createElement("button");
    b.className = "cell" + (ch === app.alphaIdx ? " alpha" : "");
    b.title = v.cap;
    const lab = document.createElement("span");
    lab.className = "clab";
    lab.textContent = ch === app.alphaIdx ? "α" : String(ch);
    b.append(makeThumb(ch), lab);
    b.onmouseenter = () => setView(ch, true);
    b.onclick = () => setView(ch);
    v.bar = b;
    chanStrip.appendChild(b);
  });
  chanStrip.onmouseleave = () => setView(app.lockedView, true);

  function setView(id, preview) {
    app.viewId = id;
    if (!preview) {
      app.lockedView = id;
      if (typeof id === "number") app.lastChan = id;
    }
    const v = findView(app.views, id);
    app.views.forEach(w => {
      if (w.btn) w.btn.classList.toggle("active", w.id === app.lockedView);
      if (w.bar) w.bar.classList.toggle("active", w.id === app.lockedView);
    });
    const cap = $("viewCap");
    if (cap) cap.textContent = v.cap;
    updateLegend(app);
  }
  app.setView = setView;

  app.stepChannel = d => {
    if (typeof app.lockedView !== "number") return;
    const i = app.chanIds.indexOf(app.lockedView);
    setView(app.chanIds[(i + d + app.chanIds.length) % app.chanIds.length]);
  };

  app.cycleMode = () => {
    const modes = app.views.filter(v => !v.chan).map(v => v.id).concat(["chan"]);
    const cur = typeof app.lockedView === "number" ? "chan" : app.lockedView;
    const next = modes[(modes.indexOf(cur) + 1) % modes.length];
    setView(next === "chan" ? app.lastChan : next);
  };

  // pointer
  const cv = $("c");
  cv.addEventListener("pointerdown", e => {
    app.ptr = hooks.canvasPos(e);
    if (spaceIsFlat(app)) {
      app.drawing = true;
      paintDamage(app, app.ptr);
      app.ripples.push({ x: app.ptr.x, y: app.ptr.y, t0: performance.now(), col: "255,111,145" });
    } else {
      app.orbiting = true;
      app.moved = false;
      app.downPos = { x: e.clientX, y: e.clientY };
    }
  });
  cv.addEventListener("pointermove", e => {
    const p = hooks.canvasPos(e);
    if (app.orbiting && app.downPos) {
      const dx = e.clientX - app.downPos.x, dy = e.clientY - app.downPos.y;
      if (Math.abs(dx) + Math.abs(dy) > 3) app.moved = true;
      app.rotY += dx * 0.008;
      app.rotX = Math.max(-1.4, Math.min(1.4, app.rotX + dy * 0.008));
      app.downPos = { x: e.clientX, y: e.clientY };
    }
    app.ptr = p;
    brushLabel();
    if (app.drawing) paintDamage(app, p);
  });
  cv.addEventListener("pointerleave", () => { app.ptr = null; brushLabel(); });
  window.addEventListener("pointerup", () => {
    if (app.orbiting && !app.moved && app.ptr) {
      paintDamage(app, app.ptr);
      app.ripples.push({ x: app.ptr.x, y: app.ptr.y, t0: performance.now(), col: "255,111,145" });
    }
    app.drawing = false;
    app.orbiting = false;
    app.downPos = null;
  });

  // edge-cut buttons if present
  const cutBtn = (id, fn) => { const b = $(id); if (b) b.onclick = fn; };
  cutBtn("cutRandom", () => applyEdgeCut(app, 0.2));
  cutBtn("restoreEdges", () => applyEdgeRestore(app));

  // re-render charts when details open
  document.querySelectorAll("details").forEach(d =>
    d.addEventListener("toggle", () => {
      renderWeights(app);
      hooks.drawTrace?.();
    }));

  setView("rgb");
  setLayout("space");
  setSpeedIdx(SPEEDS.length - 1);
  brushLabel();
  renderWeights(app);

  return { setRunning, setSpeedIdx, setView, setLayout, brushLabel };
}
