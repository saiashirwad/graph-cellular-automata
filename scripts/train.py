"""Train a Graph NCA to grow a heart on a random graph, from one seed node.

    uv run python -u scripts/train.py                # train (~2 h on M-series)
    uv run python -u scripts/train.py --steps 500    # quick smoke run
    uv run python scripts/train.py --animate         # rollout gif from checkpoint

Checkpoints -> runs/, gifs -> docs/media/, metrics -> trackio.
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F

from gnca import (
    GraphNCA,
    alive_mask,
    heart_target,
    random_geometric_graph,
    ring_target,
    seed_state,
    watts_strogatz_graph,
)
from gnca import damage as dmg

p = argparse.ArgumentParser()
p.add_argument("--nodes", type=int, default=1024)
p.add_argument("--k", type=int, default=8, help="nearest neighbors per node")
p.add_argument("--channels", type=int, default=16)
p.add_argument("--steps", type=int, default=8000)
p.add_argument("--pool", type=int, default=512, help="sample pool size")
p.add_argument("--batch", type=int, default=8)
p.add_argument("--damage", type=int, default=3,
               help="samples per batch to damage (0 = off). The best-scoring "
                    "samples are hit, so the rule learns to heal a grown pattern.")
p.add_argument("--horizon", type=int, nargs=2, default=[48, 80],
               help="rollout length range; healing needs longer than growing")
p.add_argument("--noise", type=float, default=0.02, help="state noise on x0")
p.add_argument("--overflow", type=float, default=1.0,
               help="penalty on state magnitude outside [-1, 1]")
p.add_argument("--tag", default="", help="suffix for checkpoint/gif filenames")
p.add_argument("--graph", choices=["rgg", "ws"], default="rgg",
               help="rgg: random geometric graph + heart target; "
                    "ws: Watts-Strogatz small-world ring + rainbow target")
p.add_argument("--beta", type=float, default=0.05, help="WS rewiring probability")
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--animate", action="store_true", help="skip training, render checkpoint")
p.add_argument("--track", default=True, action=argparse.BooleanOptionalAction,
               help="log curves to trackio (`trackio show` to view)")
args = p.parse_args()

suffix = ("" if args.graph == "rgg" else "_ws") + args.tag
os.makedirs("runs", exist_ok=True)
os.makedirs("docs/media", exist_ok=True)
CKPT = f"runs/checkpoint{suffix}.pt"
GIF, TARGET_PNG = f"docs/media/growth{suffix}.gif", f"docs/media/target{suffix}.png"

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

track = None
if args.track and not args.animate:
    import trackio as track
    track.init(project="gnca", name=f"{args.graph}{suffix or '_base'}", config=vars(args))

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

# -------------------------------------------------------------- damage -----
pattern = dmg.pattern_nodes(target_np)
rng = np.random.default_rng(1)


def damage(xb, n):
    """Wipe all channels of some nodes in the last n samples of the batch.

    One sample gets scattered damage, the rest get balls covering 20-50% of
    the pattern.
    """
    for i in range(1, n + 1):
        nodes = (dmg.ball(pos, pattern, frac=rng.uniform(0.2, 0.5), rng=rng)
                 if i > 1 else dmg.scatter(pattern, rng=rng))
        xb[-i, torch.from_numpy(nodes).to(device)] = 0.0


PROBE_BALL = None  # fixed blob, so the probe is comparable across steps and runs


def heal_probe(grow=80, heal=160):
    """Grow, punch a hole, heal. This is the number the whole recipe chases:
    training loss can fall while regeneration stays broken."""
    global PROBE_BALL
    if PROBE_BALL is None:
        PROBE_BALL = torch.from_numpy(dmg.ball(
            pos, pattern, frac=0.25, center=int(pattern[len(pattern) // 2]))).to(device)
    with torch.no_grad():
        x = seed_state(1, N, args.channels, device, center)[0]
        for _ in range(grow):
            x = model(x, edges) * alive_mask(x, edges, N)
        grown = ((x[:, :4] - target) ** 2).mean().item()
        x[PROBE_BALL] = 0.0
        for _ in range(heal):
            x = model(x, edges) * alive_mask(x, edges, N)
        return grown, ((x[:, :4] - target) ** 2).mean().item()


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
        xb = pool[idx].clone()                               # (B, N, C)

        # rank the batch worst-first, so xb[0] is the least-formed pattern and
        # xb[-1] the most. Damage lands on the best ones: healing a grown heart
        # is the behavior we want, and gradient on an already-broken state is
        # spent twice over.
        with torch.no_grad():
            start_loss = ((xb[..., :4] - target) ** 2).mean((1, 2))
        order = start_loss.argsort(descending=True)
        xb, idx = xb[order], idx[order.cpu()]                # idx lives on cpu

        if step % 8 == 0:  # periodically train from the bare seed (long horizons)
            xb[0] = seed_state(1, N, args.channels, device, center)[0]
        if args.damage:
            damage(xb, args.damage)

        x0 = xb.reshape(-1, args.channels)
        if args.noise:  # widens the basin and keeps gradients from diverging
            x0 = x0 + args.noise * torch.randn_like(x0)

        x = run_ca(x0, n_steps=np.random.randint(args.horizon[0], args.horizon[1] + 1))
        mse = F.mse_loss(x.view(args.batch, N, -1)[..., :4], target.expand(args.batch, N, 4))
        loss = mse
        if args.overflow:  # states that run away are the usual failure after damage
            loss = loss + args.overflow * (x - x.clamp(-1, 1)).abs().mean()

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # stability
        opt.step()

        with torch.no_grad():  # write back to pool; worst sample -> fresh seed
            out = x.view(args.batch, N, -1).detach()
            per_sample = ((out[..., :4] - target) ** 2).mean((1, 2))
            out[per_sample.argmax()] = seed_state(1, N, args.channels, device, center)[0]
            pool[idx] = out
        if track and step % 20 == 0:
            track.log({"loss": loss.item(), "mse": mse.item()}, step=step)
        if step % 200 == 0:
            print(f"step {step:6d}  loss {loss.item():.6f}  mse {mse.item():.6f}")
        if step % 1000 == 0:
            grown, healed = heal_probe()
            print(f"    probe: grown {grown:.4f}  healed {healed:.4f}")
            if track:
                track.log({"probe_grown": grown, "probe_healed": healed}, step=step)
        if step % 2000 == 0:
            torch.save({"model": model.state_dict(), "pos": pos, "target": target.cpu(),
                        "edges": edges.cpu(), "channels": args.channels}, CKPT)

    torch.save({"model": model.state_dict(), "pos": pos, "target": target.cpu(),
                "edges": edges.cpu(), "channels": args.channels}, CKPT)
    print(f"saved {CKPT}")
    if track:
        track.finish()

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
