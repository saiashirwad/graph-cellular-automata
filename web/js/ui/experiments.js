import { seed, clearAll, paintDamage, softSigh, applyEdgeRestore } from "../model.js";
import { alivePct } from "../render/draw.js";
import { spaceIsFlat } from "../layout/index.js";

function woundRandom(app) {
  const alive = [];
  for (let i = 0; i < app.n; i++)
    if (app.x[i * app.c + app.alphaIdx] > 0.1) alive.push(i);
  if (!alive.length) return;
  const ci = alive[(Math.random() * alive.length) | 0];
  const cx0 = app.dispPos[ci * 2], cy0 = app.dispPos[ci * 2 + 1];
  const prev = app.brushR;
  app.brushR = 0.12;
  paintDamage(app, { x: cx0, y: cy0 });
  app.brushR = prev;
  softSigh(app, cx0, cy0);
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
