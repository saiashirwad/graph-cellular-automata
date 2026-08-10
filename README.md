# Graph cellular automata

Distill's [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca),
but on a graph instead of a grid. One seed node grows into the whole pattern
using only messages between neighbors.

Live demo: **[gca.texoport.in](https://gca.texoport.in/)** — Stanford bunny on a
surface point cloud (drag to orbit, click to wound).

| Target | Rollout |
| ------ | ------- |
| ![bunny target](docs/media/target_bunny_pc.png) | ![bunny growth](docs/media/growth_bunny_pc.gif) |

Every node sits on the mesh. Colour comes from surface normals, so a regrown
ear comes back in a colour you can check. Damage training fills holes after
wounds (eval MSE after heal ≈ 0.015 on band / ball / scatter / half cuts).

## How it works

A grid CA is already a GNN with a fixed lattice and a hand-written rule.
Relax both:

| | neighborhood | rule |
| --- | --- | --- |
| classic CA | fixed grid | hand-written |
| Neural CA (Distill) | fixed grid (Sobel) | learned MLP |
| Graph NCA (this) | any graph (message passing) | learned MLP |

Each node holds 16 numbers: RGB, alpha, and 12 hidden channels. Every step it
sees its state, the mean of its neighbors, and the mean difference to them. A
small shared MLP updates the state. About half the nodes skip each step so
nothing can depend on a global clock.

Training is the growing-NCA recipe: sample pool, random-length rollouts, MSE on
RGBA, alive-masking by max-pooling alpha over neighbors, plus damage on the
best samples so the rule learns to heal.

Related: [Learning Graph Cellular Automata](https://arxiv.org/abs/2110.14237),
[Mesh Neural CA](https://meshnca.github.io/),
[E(n)-equivariant GNCA](https://arxiv.org/abs/2301.10497).

## Run it

```sh
uv sync

# surface bunny (what the demo ships)
uv run python scripts/fetch_pointclouds.py          # once: sample meshes → data/pointclouds/
uv run python -u scripts/train.py \
  --target bunny --nodes 1536 --k 12 \
  --horizon 64 120 --damage 3 --steps 8000 --tag _pc

uv run python scripts/train.py --animate --target bunny --tag _pc
uv run python scripts/eval_damage.py runs/checkpoint_bunny_pc.pt --grow 160 --heal 200
uv run python scripts/export_web.py runs/checkpoint_bunny_pc.pt \
  --out web/bundle_bunny.js --var BUNDLE_BUNNY
open web/index.html
```

`--animate` reloads the checkpoint as-is (positions, edges, seed). You do not
need matching `--nodes` or the raw meshes just to render a gif.

Keep `-u` if you redirect logs. Watch `probe_healed` more than loss — loss can
fall while regeneration stays broken.

Other shapes after fetch: `--target spot|teapot|armadillo` with similar knobs
(`k` 12–16, longer horizon on thin topology). 2-d patterns still work
(`heart`, `star`, `annulus`, `lobes`, `ring`).

## Healing

Growing and healing are different skills. Train only to grow and a finished
pattern has a narrow basin: punch a hole and it may stall or rot. So every step
sort the batch by loss and wreck the best samples. The basin that matters is
around a finished pattern.

Bunny (`_pc`), after 160 grow / 200 heal steps:

| damage | after hit (alive / mse) | healed t+200 |
| ------ | ----------------------- | ------------ |
| band | 80% / 0.10 | **100% / 0.015** |
| ball, 25% | 75% / 0.13 | **100% / 0.016** |
| scatter 25% | 75% / 0.12 | **100% / 0.015** |
| right half | 53% / 0.21 | **100% / 0.017** |

## Deploy

Static site is `web/` (Cloudflare Pages):

```sh
npx wrangler pages deploy web --project-name graph-cellular-automata
```

## Things to try

- more surface clouds (`spot`, `teapot`, …) in the demo picker
- cut edges during training, not only at eval
- train on a fresh random graph every episode
- swap `heart_target` / bake your own point cloud

The rule does not learn “the bunny.” It learns the bunny **on this graph**.
Thin the edges and the pattern degrades before any node has died. A grid CA
cannot fail that way, and that is the interesting part.
