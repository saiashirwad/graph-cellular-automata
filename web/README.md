# Web demo

Browser exhibit for a trained Graph Neural CA. Static files only — no app
server, no framework. Cloudflare Pages (or any static host) is enough.

## Run locally

```sh
open web/index.html          # file:// works — demo.js is one classic bundle
# or: cd web && python -m http.server 8000
```

`?cpu` forces the JS engine. `?model=bunny` selects a registered model.

## Layout

```
web/
  index.html         shell markup
  css/demo.css       styles
  js/                source (edit here)
    main.js          boot + rAF loop
    ca/              CPU step + WebGPU backend + Engine interface
    graph/ layout/ render/ ui/
  artifacts/         generated ESM (weights, pca, umap)
  demo.js            checked-in bundle of js/ + artifacts/ + typegpu
  package.json       esbuild + typegpu
```

One package, one `node_modules`, one build:

```sh
cd web
npm install
npm run build          # → demo.js
npm run watch          # rebuild on edit
```

## Rebuild model artifacts

```sh
uv run python scripts/export_web.py runs/checkpoint_bunny_pc.pt \
  --out web/artifacts/bunny.js
uv run python scripts/render/umap_states.py \
  --ckpt runs/checkpoint_bunny_pc.pt \
  --out-js web/artifacts/bunny-umap.js
cd web && npm run build
```

Register another target in `js/models.js`.

## Deploy

```sh
cd web && npm run deploy
```

That rebuilds `demo.js`, then uploads only the runtime files (`index.html`,
`css/`, `demo.js`) via Wrangler Pages. Source under `js/`, `artifacts/`, and
`node_modules` are excluded by `.cfignore`.

## Rule parity

Browser step matches training:

`[self, mean(neighbors), gated_mean_diff, log1p(deg)]`

```sh
uv run python scripts/test_web_rule.py
```
