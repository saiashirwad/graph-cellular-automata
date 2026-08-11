"""Measure the graph-native failure mode: edge thinning vs. node damage.

The claim in the README is that the rule learns the bunny *on this graph*:
cut edges and the pattern degrades even though no node has died. This scores
that claim as a curve. For every damage fraction, two channels:

  edge  - remove a random fraction of ALL edges (states untouched; the
          wound is purely topological). The cut persists through healing.
  node  - zero a random fraction of PATTERN nodes (the control; nodes stay
          in the graph and must regrow from neighbors).

Same fractions, same seeds, same stochastic update draws up to the hit, so
the two curves are directly comparable. The interesting number is healed MSE
at equal fractions: node damage recovers (the old table says 25% -> 0.015),
edge damage is the thing a grid CA cannot feel.

    uv run python scripts/eval_edges.py                 # full sweep, ~2 min
    uv run python scripts/eval_edges.py --fracs 0 .3 .5 --seeds 2 --png ""
Writes runs/edge_vs_node.csv and docs/media/edge_vs_node.png (mean +- 1 std).
"""
import argparse
import csv

import numpy as np
import torch

from gnca import damage as dmg
from gnca import load_checkpoint, rollout, seed_state

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument("--ckpt", default="runs/checkpoint_bunny_pc.pt")
p.add_argument("--grow", type=int, default=200, help="steps to steady state before the hit")
p.add_argument("--heal", type=int, default=200, help="steps allowed to heal after the hit")
p.add_argument("--fracs", type=float, nargs="+", default=[0, .05, .1, .15, .2, .3, .4, .5, .7],
               help="fraction removed (edges or pattern nodes)")
p.add_argument("--seeds", type=int, default=5)
p.add_argument("--csv", default="runs/edge_vs_node.csv")
p.add_argument("--png", default="docs/media/edge_vs_node.png", help="empty string to skip the plot")
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
ck = load_checkpoint(args.ckpt, device)
N, edges = ck.n, ck.edges
target = torch.from_numpy(ck.target).to(device)
if ck.pad:
    print(f"note: checkpoint percept predates the degree feature, padded {ck.pad} zero column(s)")
print(f"model: {args.ckpt} | N={N} | edges={edges.shape[1]} | grow {args.grow} / heal {args.heal}")

edges_np = ck.edges_np
pattern = dmg.pattern_nodes(ck.target)
live_target = float((ck.target[:, 3] > 0.5).mean())

def stats(x):
    alive = float((x[:, 3] > 0.1).float().mean()) / max(live_target, 1e-6) * 100
    mse = float(((x[:, :4] - target) ** 2).mean())
    return alive, mse

rows = []
for seed in range(args.seeds):
    # same stochastic draws for every (frac, channel) at this seed, up to the hit
    torch.manual_seed(seed)
    x = rollout(ck.model, seed_state(1, N, ck.channels, device, ck.center)[0],
                edges, args.grow)
    grown_alive, grown_mse = stats(x)
    print(f"seed {seed}: grown alive={grown_alive:.1f}% mse={grown_mse:.4f}")

    for frac in args.fracs:
        for ch_i, channel in enumerate(("edge", "node")):
            rng = np.random.default_rng(seed * 100000 + round(frac * 1000) * 10 + ch_i)
            # reseed torch per rollout so every (seed, frac, channel) is
            # reproducible on its own, independent of loop order
            torch.manual_seed(seed * 100000 + round(frac * 1000) * 10 + ch_i + 1)
            use_edges = edges
            hit = x.clone()
            if channel == "edge":
                keep = dmg.cut_random(edges_np, frac, rng)
                use_edges = edges[:, torch.from_numpy(keep).to(device)]
            elif frac > 0:
                hit[dmg.scatter(pattern, frac=frac, rng=rng)] = 0.0
            hit_alive, hit_mse = stats(hit)
            xh = rollout(ck.model, hit, use_edges, args.heal)
            healed_alive, healed_mse = stats(xh)
            rows.append([seed, frac, channel, round(hit_mse, 4), round(hit_alive, 1),
                         round(healed_mse, 4), round(healed_alive, 1)])
            print(f"  frac={frac:.2f} {channel:4s}: hit mse={hit_mse:.4f} alive={hit_alive:.0f}% "
                  f"-> healed mse={healed_mse:.4f} alive={healed_alive:.0f}%")

with open(args.csv, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["seed", "frac", "channel", "hit_mse", "hit_alive", "healed_mse", "healed_alive"])
    w.writerows(rows)
print(f"wrote {args.csv}")

if args.png:
    import os
    os.environ.pop("MPLBACKEND", None)  # agent shells leak matplotlib_inline
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rec = {(r[0], r[1], r[2]): r[3:] for r in rows}   # (seed, frac, channel) -> metrics
    fracs = sorted({r[1] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for ch, color in (("edge", "tab:red"), ("node", "tab:blue")):
        for panel, col, log in ((0, 2, True), (1, 3, False)):
            ys = np.array([[rec[(s, f, ch)][col] for s in range(args.seeds)] for f in fracs])
            if log:
                ys = np.log10(np.maximum(ys, 1e-4))
            m, sd = ys.mean(1), ys.std(1)
            ax[panel].plot(fracs, m, "-o", color=color, label=ch)
            ax[panel].fill_between(fracs, m - sd, m + sd, alpha=0.15, color=color)
    ax[0].set_xlabel("fraction removed")
    ax[0].set_ylabel("healed MSE (RGBA, log scale)")
    ax[0].legend(title="damage channel")
    ax[1].set_xlabel("fraction removed")
    ax[1].set_ylabel("healed alive %")
    ax[1].legend(title="damage channel")
    fig.suptitle("bunny point cloud: node damage heals, edge damage rots in place")
    fig.tight_layout()
    fig.savefig(args.png, dpi=150)
    print(f"wrote {args.png}")
