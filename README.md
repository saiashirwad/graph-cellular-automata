# Graph cellular automata

Distill's [Growing Neural Cellular Automata](https://distill.pub/2020/growing-ca),
but on a random graph instead of a grid.

One seed node grows into the whole pattern, using nothing but messages passed
between neighbors.

Live demo: **[gca.texoport.in](https://gca.texoport.in/)**

| Target | Rollout |
|---|---|
| ![target](target.png) | ![growth](growth.gif) |

## Try it

```sh
uv venv && uv pip install torch numpy matplotlib pillow trackio
uv run python -u train.py --steps 10000 --tag _dmg
```

About two hours on an M-series Mac. Keep the `-u`, or the log stays empty
until the run exits.

Then render it, or open the browser demo:

```sh
uv run python train.py --animate --tag _dmg   # writes growth.gif
uv run python export_web.py                   # writes web/bundle.js
open web/index.html
```

## How it works

A grid CA is already a GNN, just with a fixed lattice and a hand-written rule.
Relax both:

| | neighborhood | rule |
|---|---|---|
| classic CA | fixed grid | hand-written |
| Neural CA (Distill) | fixed grid (Sobel conv) | learned MLP |
| Graph NCA (this repo) | any graph (message passing) | learned MLP |

Each node holds 16 numbers: RGB, alpha, and 12 hidden channels.

Every step, a node looks at three things, its own state, the mean of its
neighbors, and the mean difference to its neighbors. A small shared MLP turns
that into an update. A random mask skips about half the nodes each step, so
nothing can depend on a global clock.

The rest is the growing-NCA recipe. A sample pool, rollouts of random length,
MSE on RGBA against the target, and alive-masking by max-pooling alpha over
neighbors.

Prior work: [Learning Graph Cellular Automata](https://arxiv.org/abs/2110.14237)
(NeurIPS 2021), [Mesh Neural CA](https://meshnca.github.io/),
[E(n)-equivariant GNCA](https://arxiv.org/abs/2301.10497).

## Healing

Growing and healing turn out to be different skills.

A rule trained only to grow does reach the target, but the basin around it is
narrow. Punch a hole in a finished heart and it refills maybe two thirds of the
way, stalls, then rots.

So train on broken patterns. Every step, sort the batch by loss and wreck the
three *best* samples, zeroing all 16 channels wherever the damage lands.
Hitting the best ones is the trick. The basin worth widening is the one around
the finished pattern, and a sample that was already ruined teaches nothing.

Two shapes of damage, neither of which a grid could give you:

- a ball around a random node, covering 20 to 50% of the pattern (two samples)
- a scattered quarter of the pattern nodes, no shape at all (one sample)

Smaller things that come with it:

- rollouts of 48 to 80 steps, since healing takes longer than growing
- a pool of 512
- noise of 0.02 on the starting state
- a penalty on states drifting outside [-1, 1]

`--damage 0` turns all of it off and gives you the plain growing rule back.

The recipe is borrowed. [Growing NCA](https://distill.pub/2020/growing-ca)
damages 3 of 8 samples, [VNCA](https://arxiv.org/abs/2201.12360) damages a
quarter of the batch, [self-classifying MNIST](https://distill.pub/2020/selforg/mnist/)
adds the noise, [E(n)-GNCA](https://arxiv.org/abs/2301.10497) puts the pool on
graphs. None of them study *when* to start damaging, so this starts at step 0
like they all do.

## Watching a run

```sh
uv run trackio show
```

Watch `probe_healed`, not the loss. Loss can fall for hours while healing stays
broken, which is the whole trap. Every 1000 steps the probe grows the heart,
punches a fixed 25% hole, heals for 160 steps, and records the error.

To compare two checkpoints:

```sh
uv run python eval_damage.py checkpoint.pt checkpoint_dmg.pt
```

Four kinds of damage, a band, a ball, a scattered quarter, and one whole half.
It reseeds everything per checkpoint so you compare models and not luck. I got
that wrong at first and spent a while reading noise as signal.

## The browser demo

Running at [gca.texoport.in](https://gca.texoport.in/), or open
`web/index.html` after an export.

The forward pass is reimplemented in plain JavaScript. No PyTorch, no server,
no build step. It matches the Python model to 4.8e-7.

What you can do in it:

- drag to damage cells and watch them regrow
- cut edges instead of nodes, across the middle or at random, and watch the
  dynamics cope with a rewired graph
- switch views (`V`) to see update activity, the alpha channel, or a PCA of the
  12 hidden channels, which is as close as this gets to an X-ray
- switch layouts (`L`) between trained positions, a 3-d spectral embedding you
  can orbit, and a live PCA that arranges nodes by what they are thinking
- hover a node to see exactly which neighbors it listens to

Plus the usual reseed, pause, clear, speed and brush sliders, and a loss
sparkline.

## Files

| | |
|---|---|
| `gnca.py` | model, graphs, targets |
| `train.py` | training, damage, tracking, gif |
| `eval_damage.py` | score a checkpoint on four damage types |
| `damage.py` | slice the heart in half, animate the healing |
| `topological_damage.py` | cut edges instead of nodes |
| `hidden_channels.py` | PCA of the hidden channels |
| `export_web.py` | checkpoint to `web/bundle.js` |

## Things to try

- swap `heart_target` in `gnca.py` for your own pattern
- other graphs: grids with holes, small-world rings, trees
- max aggregation instead of mean
- cut edges during training, not just nodes, so the rule survives a rewired
  world and not only a wounded one

The result I keep coming back to is that a rule trained on one topology fails
on every other one, and fails differently on each. It never learns the heart.
It learns the heart on this graph. A grid CA cannot fail that way, and I think
it is the most interesting thing here.
