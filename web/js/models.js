/**
 * Model registry. Artifacts are static imports so the demo can be bundled
 * into one classic script that still opens from file://.
 */
import bunny from "../artifacts/bunny.js";
import bunnyPca from "../artifacts/bunny-pca.js";
import bunnyUmap from "../artifacts/bunny-umap.js";

export const MODELS = {
  bunny: {
    label: "Stanford bunny",
    bundle: bunny,
    pca: bunnyPca,
    umap: bunnyUmap,
  },
};

export async function loadModel(id = "bunny") {
  const entry = MODELS[id];
  if (!entry) throw new Error(`unknown model: ${id}`);
  return {
    bundle: entry.bundle,
    pca: entry.pca ?? null,
    umap: entry.umap ?? null,
    label: entry.label,
  };
}
