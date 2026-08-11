import { seed, clearAll, applyEdgeCut, applyEdgeRestore } from "../model.js";
import { availableLayouts } from "../layout/index.js";

export function wireKeys(app, experiments) {
  window.addEventListener("keydown", e => {
    if (e.target.tagName === "INPUT") return;
    if (e.code === "Space") {
      e.preventDefault();
      app.setRunning(!app.running);
    } else if (e.key === "r") {
      seed(app);
      app.autoFf = true;
      app.setSpeedIdx?.(8);
      app.trace?.clear();
      if (app.dispPos)
        app.ripples.push({
          x: app.dispPos[app.seed * 2],
          y: app.dispPos[app.seed * 2 + 1],
          t0: performance.now(),
          col: "255,255,255",
        });
    } else if (e.key === "k") {
      clearAll(app);
    } else if (e.key === "g") {
      app.showGhost = !app.showGhost;
    } else if (e.key === "v") {
      app.cycleMode();
    } else if (e.key === "l") {
      const layouts = availableLayouts(app);
      const i = layouts.findIndex(l => l.id === app.layoutId);
      app.setLayout(layouts[(i + 1) % layouts.length].id);
    } else if (e.key === "ArrowLeft") {
      e.preventDefault();
      app.stepChannel(-1);
    } else if (e.key === "ArrowRight") {
      e.preventDefault();
      app.stepChannel(1);
    } else if (e.key === "Escape" && typeof app.lockedView === "number") {
      app.setView("rgb");
    } else if (e.key >= "1" && e.key <= "3") {
      experiments.runExperiment(+e.key - 1);
    } else if (e.key === "e") {
      applyEdgeCut(app, 0.2);
    } else if (e.key === "E") {
      applyEdgeRestore(app);
    }
  });
}
