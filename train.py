"""Train a Graph NCA to grow a heart on a random graph, from one seed node.

    uv run python train.py                # train (~10-20 min on M-series)
    uv run python train.py --steps 500    # quick smoke run
    uv run python train.py --animate      # rollout gif from checkpoint

Checkpoints -> checkpoint.pt, snapshots -> snapshots/, gif -> growth.gif
"""
import argparse
import numpy as np
import torch
import torch.nn.functional as F

from gnca import (GraphNCA, random_geometric_graph, watts_strogatz_graph,
                  heart_target, ring_target, alive_mask, seed_state)

p = argparse.ArgumentParser()
p.add_argument("--nodes", type=int, default=1024)
p.add_argument("--k", type=int, default=8, help="nearest neighbors per node")
p.add_argument("--channels", type=int, default=16)
p.add_argument("--steps", type=int, default=8000)
p.add_argument("--pool", type=int, default=256, help="sample pool size")
p.add_argument("--batch", type=int, default=8)
p.add_argument("--graph", choices=["rgg", "ws"], default="rgg",
               help="rgg: random geometric graph + heart target; "
                    "ws: Watts-Strogatz small-world ring + rainbow target")
p.add_argument("--beta", type=float, default=0.05, help="WS rewiring probability")
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--animate", action="store_true", help="skip training, render checkpoint")
args = p.parse_args()

suffix = "" if args.graph == "rgg" else "_ws"
CKPT, GIF, TARGET_PNG = f"checkpoint{suffix}.pt", f"growth{suffix}.gif", f"target{suffix}.png"

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

# ------------------------------------------------------- graph + target ----
if args.graph == "ws":
    pos, edges = watts_strogatz_graph(args.nodes, k=args.k, beta=args.beta)
    target_np = ring_target(pos)
    seed_at = np.array([1.0, 0.5])                               # node 0 on the ring
else:
    pos, edges = random_geometric_graph(args.nodes, k=args.k)
    target_np = heart_target(pos)
    seed_at = np.array([0.5, 0.45])                              # heart center
N = pos.shape[0]
target = torch.from_numpy(target_np).to(device)                  # (N, 4)
center = int(np.argmin(((pos - seed_at) ** 2).sum(1)))           # seed node

# batched edges: replicate the graph args.batch times with node-index offsets
offsets = torch.arange(args.batch, dtype=torch.int64)[:, None] * N
bedges = (edges[None] + offsets[..., None]).permute(1, 0, 2).reshape(2, -1).to(device)
edges = edges.to(device)

model = GraphNCA(channels=args.channels).to(device)
opt = torch.optim.Adam(model.parameters(), lr=args.lr)

# ---------------------------------------------------------------- pool -----
pool = seed_state(args.pool, N, args.channels, device, center)


def run_ca(x0, n_steps):
    """Roll out the CA, with pre-step alive masking like the growing NCA."""
    x = x0
    for _ in range(n_steps):
        m = alive_mask(x, bedges if x.shape[0] == args.batch * N else edges, N)
        x = model(x, bedges if x.shape[0] == args.batch * N else edges)
        x = x * m
    return x


# ---------------------------------------------------------------- train ----
if not args.animate:
    for step in range(1, args.steps + 1):
        idx = torch.randint(0, args.pool, (args.batch,))
        x0 = pool[idx].reshape(-1, args.channels).clone()
        if step % 8 == 0:  # periodically train from the bare seed (long horizons)
            x0[:N] = seed_state(1, N, args.channels, device, center)[0]

        x = run_ca(x0, n_steps=np.random.randint(40, 65))
        loss = F.mse_loss(x.view(args.batch, N, -1)[..., :4], target.expand(args.batch, N, 4))

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
        opt.step()

        with torch.no_grad():  # write back to pool; worst sample -> fresh seed
            xb = x.view(args.batch, N, -1).detach()
            per_sample = ((xb[..., :4] - target) ** 2).mean((1, 2))
            xb[per_sample.argmax()] = seed_state(1, N, args.channels, device, center)[0]
            pool[idx] = xb
        if step % 200 == 0:
            print(f"step {step:6d}  loss {loss.item():.6f}")
        if step % 2000 == 0:
            torch.save({"model": model.state_dict(), "pos": pos, "target": target.cpu(),
                        "edges": edges.cpu(), "channels": args.channels}, CKPT)

    torch.save({"model": model.state_dict(), "pos": pos, "target": target.cpu(),
                "edges": edges.cpu(), "channels": args.channels}, CKPT)
    print(f"saved {CKPT}")

# ------------------------------------------------------------ visualize ----
import os
os.environ.pop("MPLBACKEND", None)  # don't inherit a notebook backend
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ckpt = torch.load(CKPT, weights_only=False)
model.load_state_dict(ckpt["model"])
pos, tgt = ckpt["pos"], ckpt["target"].numpy()

def draw(ax, rgba, title):
    ax.clear()
    ax.scatter(pos[:, 0], pos[:, 1], c=np.clip(rgba[:, :3], 0, 1),
               s=14, alpha=np.clip(rgba[:, 3], 0, 1), linewidths=0)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title=title)
    ax.axis("off")

fig, ax = plt.subplots(figsize=(5, 5), dpi=120)
x = seed_state(1, N, model.channels, device, center)[0]
frames = []
with torch.no_grad():
    for t in range(120):
        frames.append(x[:, :4].cpu().numpy().copy())
        m = alive_mask(x, edges, N)
        x = model(x, edges) * m

anim = FuncAnimation(fig, lambda i: draw(ax, frames[i], f"graph NCA, t={i}"),
                     frames=len(frames), interval=60)
anim.save(GIF, writer=PillowWriter(fps=16))
draw(ax, tgt, "target")
plt.savefig(TARGET_PNG)
print(f"saved {GIF} and {TARGET_PNG}")
