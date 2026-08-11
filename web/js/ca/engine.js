/** Engine interface: CPU always; WebGPU when navigator.gpu works. */

import { stepCpu } from "./step-cpu.js";
import { create as createGpuBackend } from "./gpu.js";

/**
 * @typedef {object} Engine
 * @property {(k: number) => void} step
 * @property {() => void} markStateDirty
 * @property {(off: Int32Array, src: Int32Array, deg: Float32Array) => void} markGraphDirty
 * @property {() => number} avgMs
 * @property {() => string} name
 * @property {() => void} dispose
 */

/** @returns {Engine} */
export function createCpuEngine(sim) {
  let ema = 0;
  return {
    step(k) {
      const t0 = performance.now();
      for (let i = 0; i < k; i++) stepCpu(sim);
      ema = ema * 0.9 + (performance.now() - t0) * 0.1;
    },
    markStateDirty() {},
    markGraphDirty() {},
    avgMs: () => ema,
    name: () => "js",
    dispose() {},
  };
}

/**
 * Pick WebGPU when available, else CPU. GPU init is time-bounded; failure
 * always falls back to JS so the rAF loop can start.
 * @returns {Promise<Engine>}
 */
export async function createEngine(sim, opts = {}) {
  const forceCpu = opts.forceCpu
    || (typeof location !== "undefined"
        && new URLSearchParams(location.search).has("cpu"));

  if (forceCpu || typeof navigator === "undefined" || !navigator.gpu) {
    return createCpuEngine(sim);
  }

  try {
    const gpu = await withTimeout(
      createGpuBackend({
        n: sim.n,
        c: sim.c,
        h: sim.h,
        alphaIdx: sim.alphaIdx,
        x: sim.x,
        act: sim.act,
        off: sim.off,
        src: sim.src,
        deg: sim.deg,
        w1: sim.w1,
        b1: sim.b1,
        w2: sim.w2,
        gateW: sim.gateW,
        gateB: sim.gateB,
      }),
      opts.gpuTimeoutMs ?? 4000,
      "webgpu init timed out",
    );
    return wrapGpu(sim, gpu);
  } catch (err) {
    console.warn("[gnca] webgpu unavailable, staying on js:", err);
    return createCpuEngine(sim);
  }
}

function withTimeout(promise, ms, msg) {
  return new Promise((resolve, reject) => {
    const t = setTimeout(() => reject(new Error(msg)), ms);
    promise.then(
      v => { clearTimeout(t); resolve(v); },
      e => { clearTimeout(t); reject(e); },
    );
  });
}

function wrapGpu(sim, gpu) {
  return {
    step(k) {
      if (k) {
        gpu.step(k);
        sim.t += k;
      }
    },
    markStateDirty() { gpu.markStateDirty(); },
    markGraphDirty(off, src, deg) { gpu.markGraphDirty(off, src, deg); },
    avgMs: () => gpu.avgMs,
    name: () => "webgpu",
    dispose() { gpu.dispose?.(); },
  };
}
