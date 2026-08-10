# Graph cellular automata

A small mashup of graph neural networks and Distill.pub's
[Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca). The
CA lives on a random graph instead of a grid, and grows a pattern from one
seed node.

## What it looks like

| Target | Rollout |
|---|---|
| ![target](target.png) | ![growth](growth.gif) |

On the left, the target sampled at the graph's node positions. On the right,
the trained CA growing it from a single seed node, using nothing but messages
between neighbors.

## The idea

A grid CA is a special case of a GNN, with a fixed lattice and a hand-written
message function. Relax both:

| | neighborhood | rule |
|---|---|---|
| classic CA | fixed grid | hand-written |
| Neural CA (Distill) | fixed grid (Sobel conv) | learned MLP |
| Graph NCA (this repo) | arbitrary graph (message passing) | learned MLP |

Each node holds a 16-dim state, RGB plus alpha plus 12 hidden channels.
Perception is GNN-style aggregation. Each node sees `[own state,
mean(neighbors), mean(neighbors - own)]`. A tiny shared MLP proposes a
residual update, and a random per-node mask keeps updates asynchronous, so no
node can rely on a global clock. Training follows the growing-NCA recipe: a
sample pool, random-length rollouts, MSE on RGBA against the target, and
alive-masking by max-pooling alpha over graph neighbors.

Related work: [Learning Graph Cellular Automata](https://arxiv.org/abs/2110.14237)
(Grattarola et al., NeurIPS 2021),
[Mesh Neural CA](https://meshnca.github.io/) (2023),
[E(n)-equivariant GNCA](https://arxiv.org/abs/2301.10497) (2023).

## Regeneration

Growing a pattern and healing a broken one are different skills. A model
trained only to grow does reach the target, but the basin around it is narrow.
Punch a hole in a finished heart and it refills about two thirds of the way,
stalls, then decays.

The fix comes from Distill's "learning to regenerate" experiment. Train on
broken patterns directly. Every step, training sorts the batch by starting
loss and wrecks the three *best* samples, zeroing all 16 channels on the nodes
it hits. Hitting the best ones is the whole trick. The point is to widen the
basin around the finished pattern, and gradient spent on an already-ruined
state teaches nothing.

Damage comes in two shapes, both of which make sense on a graph rather than a
grid. A ball wipes out 20 to 50% of the pattern around a random node, and
takes two of the three samples. The third gets scattered damage, a random
quarter of the pattern nodes with no shape at all.

A few smaller changes come with it. Rollouts stretch to 48-80 steps, because a
healing wavefront needs longer than a growing one. The pool grows to 512
samples. Noise of sigma 0.02 on the starting state widens the basin and keeps
gradients from blowing up. A penalty on states leaving [-1, 1] catches the
runaway that tends to follow damage. All of it switches off with `--damage 0`,
which gets you the plain growing recipe back.

Where the recipe comes from: [Growing NCA](https://distill.pub/2020/growing-ca)
damages 3 of 8 lowest-loss samples,
[VNCA](https://arxiv.org/abs/2201.12360) damages a quarter of the batch,
[Self-classifying MNIST](https://distill.pub/2020/selforg/mnist/) adds state
noise, and [E(n)-GNCA](https://arxiv.org/abs/2301.10497) puts the pool on
graphs. None of them study *when* to start damaging. They all start at step 0
with a constant fraction, so this does too.

## Run it (Apple Silicon, uses MPS)

```sh
uv venv && uv pip install torch numpy matplotlib pillow trackio

uv run python -u train.py --steps 10000 --tag _dmg   # train, roughly 2 h
uv run python train.py --animate --tag _dmg          # growth.gif from checkpoint
```

The `-u` matters. Without it Python block-buffers stdout, and a redirected log
sits empty until the run exits.

Metrics go to [trackio](https://huggingface.co/blog/trackio), which runs
locally, writes SQLite, and needs no account. Watch them in another terminal:

```sh
uv run trackio show
```

Watch `probe_healed` rather than the training loss. Every 1000 steps the probe
grows the heart, punches a fixed 25% hole, heals for 160 steps, and records the
error. Training loss can keep falling while regeneration stays broken, which is
exactly the trap this whole recipe exists to escape. `--no-track` skips all of
it.

`eval_damage.py` scores one checkpoint against another on four kinds of damage,
a horizontal band, a ball, a scattered quarter, and one whole half:

```sh
uv run python eval_damage.py checkpoint.pt checkpoint_dmg.pt
```

It reseeds both the damage regions and the CA's random updates for each
checkpoint, so the numbers compare models rather than luck. I got this wrong
the first time and spent a while reading noise as signal.

## Files

| | |
|---|---|
| `gnca.py` | the model, the graphs (random geometric, Watts-Strogatz), the targets |
| `train.py` | training, damage augmentation, tracking, rollout gif |
| `eval_damage.py` | four damage types, alive% and error through healing |
| `damage.py` | slice the grown heart in half, animate the healing |
| `topological_damage.py` | cut *edges* instead of nodes and watch it adapt |
| `hidden_channels.py` | PCA of the 12 hidden channels, the learned morphogens |
| `export_web.py` | checkpoint to `web/bundle.js` for the browser demo |

## Web demo (browser-only, no server)

After training, export the weights to JSON and open the demo:

```sh
uv run python export_web.py          # checkpoint.pt -> web/bundle.js
open web/index.html                  # or double-click it
```

The whole CA forward pass runs in plain JavaScript. No PyTorch, no server, no
build step. The canvas gives you:

- drag to damage cells, with a brush cursor and ripple, then watch them regrow
- reseed, pause, clear (`R`, `Space`, `K`)
- view modes (`V`): visible RGB, update activity showing where the rule fires,
  a PCA projection of the hidden channels, the alpha channel, and each of the
  12 hidden channels on its own
- topological damage: cut every edge across the middle, or drop 20% at random,
  with `E` to undo
- layouts (`L`): the trained 2-d positions, a 3-d spectral embedding of the
  bare topology that you can drag to orbit, or a live PCA of hidden states that
  clusters nodes by internal chemistry
- hover a node to see which neighbors it listens to
- a strip chart of the seed node's 16 channels over time, and heatmaps of the
  rule's two weight matrices
- speed and brush-radius sliders, toggles for the ghost target, edges, and glow
- step, alive and loss stats, with a loss sparkline

The JS forward pass matches the PyTorch model to 4.8e-7 max absolute error.

## Things to try

- swap `heart_target` in `gnca.py` for your own pattern
- other graphs: grids with missing edges, small-world rings, trees
- max aggregation instead of mean, or edge features carrying relative position
- edge-cut augmentation during training, dropping 10-30% of one sample's edges
  each step, so the rule survives a rewired world and not only a wounded one

The result I keep coming back to is that a rule trained on one topology fails
on every other one, and fails differently on each. It doesn't learn the heart.
It learns the heart on this graph. That's a failure mode a grid CA cannot have,
and it's the most interesting thing here.
