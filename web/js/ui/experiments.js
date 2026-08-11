import { seed, clearAll, paintDamage, applyEdgeRestore } from "../model.js";
import { alivePct } from "../render/draw.js";
import { spaceIsFlat } from "../layout/index.js";

function woundRandom(app) {
  const alive = [];
  for (let i = 0; i < app.n; i++)
    if (app.x[i * app.c + app.alphaIdx] > 0.1) alive.push(i);
  if (!alive.length) return;
  const ci = alive[(Math.random() * alive.length) | 0];
  const cx0 = app.dispPos[ci * 2], cy0 = app.dispPos[ci * 2 + 1];
  for (let i = 0; i < app.n; i++) {
    const dx = app.dispPos[i * 2] - cx0, dy = app.dispPos[i * 2 + 1] - cy0;
    if (dx * dx + dy * dy < 0.12 * 0.12)
      for (let ch = 0; ch < app.c; ch++) app.x[i * app.c + ch] = 0;
  }
  app.engine?.markStateDirty();
  app.ripples.push({ x: cx0, y: cy0, t0: performance.now(), col: "255,111,145" });
}

export function createExperiments(app) {
  const storyEl = document.getElementById("story");
  const setStory = html => { storyEl.innerHTML = html; };

  const EXPERIMENTS = [
    {
      num: "01", name: "Grow", desc: "one seed cell grows the pattern",
      story: "One seed cell, one shared rule. Each cell sees only its neighbors.",
      run() {
        app.setLayout("space");
        app.setView("rgb");
        applyEdgeRestore(app);
        seed(app);
        app.autoFf = true;
        app.setSpeedIdx?.(8);
        app.setRunning(true);
      },
    },
    {
      num: "02", name: "Tear", desc: "tear a hole, watch it heal",
      story: () => "The bright cells are the rule firing, rebuilding the hole — " +
        (spaceIsFlat(app) ? "drag" : "click") + " to tear your own.",
      run() {
        app.setLayout("space");
        app.setView("act");
        app.setRunning(true);
        if (alivePct(app) < 10) {
          seed(app);
          app.autoFf = true;
          app.setSpeedIdx?.(8);
          app.pendingWound = 1;
        } else {
          woundRandom(app);
        }
      },
    },
    {
      num: "03", name: "Kill", desc: "kill every cell at once",
      story: "Everything dead — and it stays dead. A cell only wakes if it or a neighbor is alive.",
      run() {
        app.setLayout("space");
        app.setView("rgb");
        clearAll(app);
        app.setRunning(true);
      },
    },
  ];

  function runExperiment(k) {
    const ex = EXPERIMENTS[k];
    ex.run();
    setStory(typeof ex.story === "function" ? ex.story() : ex.story);
  }

  const defaultStory = () => "The whole shape grew from one cell. " +
    (spaceIsFlat(app)
      ? "Drag anywhere to tear a hole — then watch it heal."
      : "Click anywhere to tear a hole — then watch it heal.");

  setStory(defaultStory());

  return {
    runExperiment,
    setStory,
    defaultStory,
    woundRandom,
    onGrown() {
      if (app.pendingWound) {
        app.pendingWound = 0;
        woundRandom(app);
      }
    },
  };
}
