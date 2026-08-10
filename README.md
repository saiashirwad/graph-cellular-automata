# Graph Cellular Automata

A minimal mashup of **graph neural networks** and Distill.pub's
[**Growing Neural Cellular Automata**](https://distill.pub/2020/growing-ca):
an NCA that lives on a random graph instead of a grid, and grows a pattern
from a single seed node.

## What it looks like

| Target | Rollout |
|---|---|
| ![target](target.png) | ![growth](growth.gif) |

The left image is the pattern sampled at the graph's node positions; the right
is the trained CA growing it from one seed node via pure neighbor message
passing. (The committed gif is from a short 600-step smoke run — re-run
`train.py --animate` after full training for the pretty version.)

## The idea

A grid CA is a special case of a GNN: fixed lattice graph + a shared,
hand-written message function. Combine the two classical relaxations:

| | neighborhood | rule |
|---|---|---|
| classic CA | fixed grid | hand-written |
| Neural CA (Distill) | fixed grid (Sobel conv) | learned MLP |
| **Graph NCA (this repo)** | **arbitrary graph (message passing)** | **learned MLP** |

Each node holds a 16-dim state (RGB + alpha + 12 hidden channels). Perception
is GNN-style aggregation: each node sees `[own state, mean(neighbors),
mean(neighbors - own)]`, a tiny shared MLP proposes a residual update, and a
stochastic per-node mask keeps updates asynchronous. Training follows the
growing-NCA recipe: a sample pool, random 40-64 step rollouts, MSE loss on
RGBA against the target, alive-masking via max-pooling over graph neighbors.

Related work: [Learning Graph Cellular Automata](https://arxiv.org/abs/2110.14237)
(Grattarola et al., NeurIPS 2021),
[Mesh Neural CA](https://meshnca.github.io/) (2023),
[E(n)-equivariant GNCA](https://arxiv.org/abs/2301.10497) (2023).

## Run it (Apple Silicon, uses MPS)

```sh
uv venv && uv pip install torch numpy matplotlib pillow
uv run python train.py              # ~8k steps, prints loss, saves checkpoint
uv run python train.py --animate    # renders growth.gif from checkpoint
```

Knobs to play with are argparse flags at the top of `train.py`
(`--nodes`, `--k`, `--channels`, ...). Things to try:

- change `heart_target` in `gnca.py` to your own pattern
- different graphs: grids with missing edges, small-world (Watts-Strogatz), trees
- swap mean aggregation for max, or add edge features (relative position)
- damage test: kill half the nodes mid-rollout and watch it regenerate
