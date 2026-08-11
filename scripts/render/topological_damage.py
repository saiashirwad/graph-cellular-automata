"""Topological damage test: grow the pattern, then cut edges and watch it adapt.

    uv run python scripts/render/topological_damage.py [--mode spatial|random] [--cut 0.3]
Saves topological_damage.gif.

This is the graph-native version of the pixel-damage demo: instead of killing
nodes, we sever edges. The pattern must cope with a rewired neighborhood.
"""
import argparse
import os

os.environ.pop("MPLBACKEND", None)
import numpy as np
import torch

from gnca import alive_mask, load_checkpoint, seed_state
from gnca import damage as dmg

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/checkpoint.pt")
p.add_argument("--out", default="docs/media/topological_damage.gif")
p.add_argument("--mode", choices=["spatial", "random"], default="spatial",
               help="spatial: cut edges crossing a horizontal line; "
                    "random: delete a fraction of all edges")
p.add_argument("--cut", type=float, default=0.4,
               help="spatial: line y-position; random: fraction of edges to remove")
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
ck = load_checkpoint(args.ckpt, device)
pos, edges, model = ck.pos, ck.edges, ck.model
C, N = ck.channels, ck.n
target = torch.from_numpy(ck.target).to(device)

GROW, ADAPT = 80, 160

# ---- determine which edges to cut ----
edges_np = ck.edges_np
if args.mode == "spatial":
    keep_mask = dmg.cut_across(pos, edges_np, args.cut)
    cut_desc = f"cut all edges crossing y={args.cut:.2f}"
else:
    keep_mask = dmg.cut_random(edges_np, args.cut, np.random.default_rng(42))
    cut_desc = f"removed {args.cut:.0%} of edges"
n_cut = (~keep_mask).sum()
print(f"{cut_desc}: {n_cut} of {edges_np.shape[1]} edges removed "
      f"({n_cut/edges_np.shape[1]:.1%})")

edges_full = edges
edges_cut = edges[:, torch.from_numpy(keep_mask).to(device)]

# ---- rollout ----
x = seed_state(1, N, C, device, ck.center)[0]
frames = []
with torch.no_grad():
    for t in range(GROW + ADAPT):
        frames.append(x[:, :4].cpu().numpy().copy())
        use_edges = edges_full if t < GROW else edges_cut
        m = alive_mask(x, use_edges)
        x = model(x, use_edges) * m
        if t == GROW:
            print(f"t={t}: switched to damaged topology ({n_cut} edges cut)")
        if t % 40 == 0:
            alive = (x[:, 3] > 0.1).float().mean().item() * 100
            loss = ((x[:, :4] - target) ** 2).mean().item()
            tag = "FULL graph" if t < GROW else "DAMAGED graph"
            print(f"t={t:4d} [{tag:13s}]  alive={alive:.0f}%  loss={loss:.4f}")

# ---- animate ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fig, ax = plt.subplots(figsize=(5, 5), dpi=120)

def draw(i):
    rgba = frames[i]
    ax.clear()
    ax.scatter(pos[:, 0], pos[:, 1], c=np.clip(rgba[:, :3], 0, 1), s=14,
               alpha=np.clip(rgba[:, 3], 0, 1), linewidths=0)
    if i < GROW:
        title = f"growing (t={i})"
        if args.mode == "spatial":
            ax.axhline(args.cut, color="red", alpha=0.3, linestyle="--")
    elif i == GROW:
        title = f"*** EDGES CUT *** (t={i})"
        if args.mode == "spatial":
            ax.axhline(args.cut, color="red", alpha=0.5, linestyle="-", linewidth=2)
    else:
        title = f"adapting (t={i})"
        if args.mode == "spatial":
            ax.axhline(args.cut, color="red", alpha=0.3, linestyle="--")
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title=title)
    ax.axis("off")

anim = FuncAnimation(fig, draw, frames=len(frames), interval=60)
anim.save(args.out, writer=PillowWriter(fps=16))
print(f"saved {args.out} ({len(frames)} frames)")
