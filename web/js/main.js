/**
 * Graph Neural Cellular Automata — browser demo entry.
 * Static ES modules; serve web/ over HTTP (Pages, or python -m http.server).
 */
import { loadModel } from "./models.js";
import { applyBundle, seed } from "./model.js";
import { createEngine } from "./ca/engine.js";
import { updateLayout, spaceIsFlat } from "./layout/index.js";
import { createRenderer, loss, alivePct } from "./render/draw.js";
import { createTrace, createSpark, renderWeights } from "./render/charts.js";
import { wireControls, SPEEDS } from "./ui/controls.js";
import { createExperiments } from "./ui/experiments.js";
import { wireKeys } from "./ui/keys.js";

function createAppShell() {
  return {
    // model (filled by applyBundle)
    n: 0, c: 0, h: 128, dim: 2, alphaIdx: 3,
    pos: null, off: null, src: null, deg: null,
    w1: null, b1: null, w2: null, gateW: null, gateB: null,
    tgt: null, seed: 0,
    x: null, act: null, t: 0,
    // display
    layoutId: "space",
    viewId: "rgb",
    lockedView: "rgb",
    lastChan: 3,
    showGhost: true,
    showEdges: true,
    glow: true,
    rotY: 0.5,
    rotX: 0.35,
    autoSpin: false,
    brushR: 0.08,
    // sim control
    running: true,
    stepsPerFrame: SPEEDS[SPEEDS.length - 1],
    stepAcc: 0,
    autoFf: true,
    spdGlide: 0,
    pendingWound: 0,
    // interaction
    ptr: null,
    drawing: false,
    orbiting: false,
    moved: false,
    downPos: null,
    ripples: [],
    // optional artifacts / engine
    pca: null,
    umap: null,
    engine: null,
    views: [],
    chanIds: [],
  };
}

async function main() {
  const app = createAppShell();
  const cv = document.getElementById("c");
  const renderer = createRenderer(cv);

  const modelId = new URLSearchParams(location.search).get("model") || "bunny";
  const loaded = await loadModel(modelId);
  applyBundle(app, loaded.bundle, { pca: loaded.pca, umap: loaded.umap });

  if (app.dim > 2) { app.rotY = 0.55; app.rotX = 0.35; }

  app.engine = await createEngine(app);

  const spark = createSpark();
  const trace = createTrace(() => app.seed, () => app.c);
  app.trace = trace;

  const hooks = {
    canvasPos: renderer.canvasPos,
    drawTrace: () => trace.draw(),
    onSeed() {
      seed(app);
      app.autoFf = true;
      app.setSpeedIdx(SPEEDS.length - 1);
      trace.clear();
      spark.clear();
      if (app.dispPos)
        app.ripples.push({
          x: app.dispPos[app.seed * 2],
          y: app.dispPos[app.seed * 2 + 1],
          t0: performance.now(),
          col: "255,255,255",
        });
    },
  };

  wireControls(app, hooks);
  const experiments = createExperiments(app);
  wireKeys(app, experiments);

  // seed ripple
  if (app.dispPos)
    app.ripples.push({
      x: app.dispPos[app.seed * 2],
      y: app.dispPos[app.seed * 2 + 1],
      t0: performance.now(),
      col: "255,255,255",
    });

  const stT = document.getElementById("stT");
  const stA = document.getElementById("stA");
  const stL = document.getElementById("stL");
  const stE = document.getElementById("stE");
  const stEng = document.getElementById("stEng");

  const grown = L => L < 0.025 || app.t > 400;
  let frame = 0, thumbTick = 0;

  function loop(now) {
    if (app.running) {
      app.stepAcc += app.stepsPerFrame;
      const k = Math.floor(app.stepAcc);
      app.stepAcc -= k;
      if (k) app.engine.step(k);
    }
    if (!spaceIsFlat(app) && app.autoSpin && !app.orbiting) app.rotY += 0.0035;
    updateLayout(app);
    renderer.draw(app, now);

    if ((frame++ & 3) === 0) {
      const L = loss(app), A = alivePct(app);
      if (app.running) {
        spark.push(L);
        trace.push(app.x);
        trace.draw();
      }
      if ((thumbTick++ & 3) === 0) renderer.paintThumbs(app);
      if (app.pendingWound && grown(L)) experiments.onGrown();
      if (app.autoFf && grown(L)) {
        app.autoFf = false;
        app.glideSpeedTo(0);
      }
      stT.textContent = app.t;
      stA.textContent = A.toFixed(0) + "%";
      stL.textContent = L.toFixed(4);
      stE.textContent = app.src.length;
      stEng.textContent = app.engine.name() + " " + app.engine.avgMs().toFixed(1) + "ms";
      spark.draw();
    }
    requestAnimationFrame(loop);
  }

  renderWeights(app);
  requestAnimationFrame(loop);
}

main().catch(err => {
  console.error(err);
  const hud = document.getElementById("hud");
  if (hud) hud.textContent = "failed to start: " + err.message;
});
