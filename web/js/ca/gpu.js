// WebGPU backend (TypeGPU + hand WGSL). Folded into demo.js by esbuild.
// ≤8 storage buffers (WebGPU default maxStorageBuffersPerShaderStage).
// Matches src/gnca/model.py: gated mean_diff + log1p(deg).
import tgpu from "typegpu";
import * as d from "typegpu/data";

const WGSL = /* wgsl */ `
struct Params {
  t : u32,
  alpha : u32,
  n : u32,
  c : u32,
  h : u32,
  // packed weight layout (f32 indices into 'w')
  oW1 : u32,
  oB1 : u32,
  oW2 : u32,
  oGW : u32,
  oGB : u32,
};
@group(0) @binding(0) var<storage, read>       xIn    : array<f32>;
@group(0) @binding(1) var<storage, read_write> xOut   : array<f32>;
@group(0) @binding(2) var<storage, read>       csrOff : array<u32>;
@group(0) @binding(3) var<storage, read>       csrSrc : array<u32>;
@group(0) @binding(4) var<storage, read>       w      : array<f32>; // all weights packed
@group(0) @binding(5) var<uniform>             params : Params;

fn pcg(input : u32) -> u32 {
  let state = input * 747796405u + 2891336453u;
  let word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
  return (word >> 22u) ^ word;
}

fn sigmoid(x : f32) -> f32 {
  return 1.0 / (1.0 + exp(-x));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3u) {
  let i = gid.x;
  if (i >= params.n) { return; }
  let C = params.c;
  let H = params.h;
  let base = i * C;
  let alpha = params.alpha;
  let e0 = csrOff[i];
  let e1 = csrOff[i + 1u];
  let deg = f32(e1 - e0);

  var alive = xIn[base + alpha] > 0.1;
  for (var e = e0; e < e1; e++) {
    let jb = csrSrc[e] * C;
    if (!alive && xIn[jb + alpha] > 0.1) { alive = true; }
  }

  let inv = 1.0 / max(deg, 1.0);
  let inW = 3u * C + 1u;

  var z : array<f32, 64>;
  for (var ch = 0u; ch < C; ch++) {
    var sum = 0.0;
    var sumDiff = 0.0;
    for (var e = e0; e < e1; e++) {
      let j = csrSrc[e];
      let jb = j * C;
      let diff = xIn[jb + ch] - xIn[base + ch];

      var aSrc = 0.0;
      var aDst = 0.0;
      let row = params.oGW + ch * 3u * C;
      for (var k = 0u; k < C; k++) {
        aSrc += w[row + k] * xIn[jb + k];
        aDst += w[row + C + k] * xIn[base + k];
      }
      var logit = aSrc + aDst + w[params.oGB + ch];
      let g3 = row + 2u * C;
      for (var k = 0u; k < C; k++) {
        let dd = xIn[jb + k] - xIn[base + k];
        logit += w[g3 + k] * abs(dd);
      }
      let g = 2.0 * sigmoid(logit);
      sum += xIn[jb + ch];
      sumDiff += g * diff;
    }
    let selfv = xIn[base + ch];
    z[ch] = selfv;
    z[C + ch] = sum * inv;
    z[2u * C + ch] = sumDiff * inv;
  }
  z[3u * C] = log(1.0 + deg);

  var harr : array<f32, 128>;
  for (var k = 0u; k < H; k++) {
    var acc = w[params.oB1 + k];
    let wb = params.oW1 + k * inW;
    for (var p = 0u; p < inW; p++) { acc += w[wb + p] * z[p]; }
    harr[k] = max(acc, 0.0);
  }

  let mask = select(0.0, 1.0, f32(pcg(i ^ pcg(params.t))) * (1.0 / 4294967296.0) < 0.5);

  var dsq = 0.0;
  for (var ch = 0u; ch < C; ch++) {
    var upd = 0.0;
    let wb = params.oW2 + ch * H;
    for (var k = 0u; k < H; k++) { upd += w[wb + k] * harr[k]; }
    let xn = select(0.0, xIn[base + ch] + upd * mask, alive);
    xOut[base + ch] = xn;
    let dd = xn - xIn[base + ch];
    dsq += dd * dd;
  }

  let ab = params.n * C;
  xOut[ab + i] = 0.85 * xIn[ab + i] + 0.15 * sqrt(dsq);
}
`;

/** Pack w1, b1, w2, gateW, gateB into one Float32Array; return {data, offsets}. */
function packWeights(w1, b1, w2, gateW, gateB) {
  const oW1 = 0;
  const oB1 = oW1 + w1.length;
  const oW2 = oB1 + b1.length;
  const oGW = oW2 + w2.length;
  const oGB = oGW + gateW.length;
  const data = new Float32Array(oGB + gateB.length);
  data.set(w1, oW1);
  data.set(b1, oB1);
  data.set(w2, oW2);
  data.set(gateW, oGW);
  data.set(gateB, oGB);
  return { data, oW1, oB1, oW2, oGW, oGB };
}

export async function create(opts) {
  const {
    n, c, h, alphaIdx, x, act, off, src,
    w1, b1, w2, gateW, gateB,
  } = opts;
  const inW = 3 * c + 1;
  if (w1.length !== h * inW || b1.length !== h || w2.length !== c * h)
    throw new Error(`weight shapes do not match C=${c}, H=${h}, inW=${inW}`);
  if (gateW.length !== c * 3 * c || gateB.length !== c)
    throw new Error("gate weight shapes do not match C");

  const packed = packWeights(w1, b1, w2, gateW, gateB);

  const root = await tgpu.init({
    device: {
      // belt-and-suspenders; packing already keeps us at 5 storage buffers
      requiredLimits: { maxStorageBuffersPerShaderStage: 8 },
    },
  });
  const device = root.device;
  const queue = device.queue;

  // surface async GPU validation errors as thrown exceptions
  device.pushErrorScope("validation");

  const stateLen = n * c + n;
  const state = [0, 1].map(() =>
    root.createBuffer(d.arrayOf(d.f32, stateLen)).$usage("storage"));
  const stateRaw = state.map(b => root.unwrap(b));

  const staticBuf = (schema, data) =>
    root.unwrap(root.createBuffer(schema, data).$usage("storage"));

  const offB = staticBuf(
    d.arrayOf(d.u32, off.length),
    new Uint32Array(off.buffer, off.byteOffset, off.length),
  );
  const srcB = staticBuf(
    d.arrayOf(d.u32, src.length),
    new Uint32Array(src.buffer, src.byteOffset, src.length),
  );
  const wB = staticBuf(d.arrayOf(d.f32, packed.data.length), packed.data);

  const Params = d.struct({
    t: d.u32, alpha: d.u32, n: d.u32, c: d.u32, h: d.u32,
    oW1: d.u32, oB1: d.u32, oW2: d.u32, oGW: d.u32, oGB: d.u32,
  });
  const paramVals = {
    t: 0, alpha: alphaIdx, n, c, h,
    oW1: packed.oW1, oB1: packed.oB1, oW2: packed.oW2,
    oGW: packed.oGW, oGB: packed.oGB,
  };
  const params = root.createUniform(Params, paramVals);
  const paramsRaw = root.unwrap(params.buffer);

  const module = device.createShaderModule({ code: WGSL });
  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module, entryPoint: "main" },
  });

  const err = await device.popErrorScope();
  if (err) {
    root.destroy();
    throw new Error(`webgpu pipeline: ${err.message}`);
  }

  const groups = [0, 1].map(p => device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: stateRaw[p] } },
      { binding: 1, resource: { buffer: stateRaw[1 - p] } },
      { binding: 2, resource: { buffer: offB } },
      { binding: 3, resource: { buffer: srcB } },
      { binding: 4, resource: { buffer: wB } },
      { binding: 5, resource: { buffer: paramsRaw } },
    ],
  }));

  const stateBytes = stateLen * 4;
  const stages = [0, 1, 2].map(() => ({
    busy: false,
    buf: device.createBuffer({
      size: stateBytes,
      usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST,
    }),
  }));

  const engine = {
    t: 0,
    cur: 0,
    avgMs: 0,
    disposed: false,

    step(k) {
      const t0 = performance.now();
      for (let s = 0; s < k; s++) {
        params.write({ ...paramVals, t: engine.t });
        engine.t++;
        const enc = device.createCommandEncoder();
        const pass = enc.beginComputePass();
        pass.setPipeline(pipeline);
        pass.setBindGroup(0, groups[engine.cur]);
        pass.dispatchWorkgroups(Math.ceil(n / 64));
        pass.end();
        queue.submit([enc.finish()]);
        engine.cur ^= 1;
      }
      engine.readback();
      const ms = performance.now() - t0;
      engine.avgMs = engine.avgMs * 0.9 + ms * 0.1;
    },

    readback() {
      const stage = stages.find(s => !s.busy);
      if (!stage) return;
      stage.busy = true;
      const enc = device.createCommandEncoder();
      enc.copyBufferToBuffer(stateRaw[engine.cur ^ 1], 0, stage.buf, 0, stateBytes);
      queue.submit([enc.finish()]);
      stage.buf.mapAsync(GPUMapMode.READ).then(() => {
        const data = new Float32Array(stage.buf.getMappedRange());
        x.set(data.subarray(0, n * c));
        act.set(data.subarray(n * c, stateLen));
        stage.buf.unmap();
        stage.busy = false;
      }).catch(() => { stage.busy = false; });
    },

    markStateDirty() {
      for (const b of stateRaw) {
        queue.writeBuffer(b, 0, x);
        queue.writeBuffer(b, n * c * 4, act);
      }
    },

    markGraphDirty(newOff, newSrc) {
      queue.writeBuffer(offB, 0, new Uint32Array(newOff.buffer, newOff.byteOffset, newOff.length));
      queue.writeBuffer(srcB, 0, new Uint32Array(newSrc.buffer, newSrc.byteOffset, newSrc.length));
    },

    dispose() {
      engine.disposed = true;
      root.destroy();
    },
  };

  engine.markStateDirty();
  return engine;
}
