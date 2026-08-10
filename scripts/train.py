"""Train a Graph NCA to grow a heart on a random graph, from one seed node.

    uv run python -u scripts/train.py                # train (~2 h on M-series)
    uv run python -u scripts/train.py --steps 500    # quick smoke run
    uv run python scripts/train.py --animate         # rollout gif from checkpoint

Checkpoints -> runs/, gifs -> docs/media/, metrics -> trackio.
Use --run NAME per experiment (baseline, deg, deg+gate, ...) so each run gets
its own checkpoint and a comparable trackio curve; `trackio show` to view.
"""
import argparse
import os
import time

import numpy as np
import torch
import torch.nn.functional as F

from gnca import (
    GraphNCA,
    alive_mask,
    load_rule,
    knn_graph,
    random_geometric_graph,
    seed_state,
    watts_strogatz_graph,
)
from gnca import damage as dmg
from gnca.pointclouds import POINTCLOUDS, load_cloud
from gnca.targets import TARGETS

ALL_TARGETS = list(TARGETS) + list(POINTCLOUDS)

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
p.add_argument("--target", default="heart", choices=ALL_TARGETS,
               help="pattern to grow. 2-d: heart/star/annulus/lobes/ring. "
                    "volume 3-d: sphere/torus/jack. surface point clouds: "
                    "bunny/spot/armadillo/teapot (run scripts/fetch_pointclouds.py first)")
p.add_argument("--graph", choices=["rgg", "ws", "auto"], default="auto",
               help="rgg: random geometric graph; ws: Watts-Strogatz ring; "
                    "auto: point-cloud k-NN for mesh targets, else rgg")
p.add_argument("--beta", type=float, default=0.05, help="WS rewiring probability")
p.add_argument("--lr", type=float, default=5e-4)
p.add_argument("--init-from", default="",
               help="checkpoint to warm-start from. If the saved first layer is "
                    "narrower than the model's (e.g. a pre-degree-feature percept), "
                    "pad its input columns with zeros: the loaded rule is then "
                    "functionally identical, and fine-tuning learns the new feature.")
p.add_argument("--animate", action="store_true", help="skip training, render checkpoint")
p.add_argument("--run", default="",
               help="experiment name for the ablation ladder (e.g. baseline, deg, "
                    "deg+gate). Names the trackio run and, when given, the checkpoint, "
                    "so parallel experiments don't overwrite each other.")
p.add_argument("--track", default=True, action=argparse.BooleanOptionalAction,
               help="log curves to trackio (`trackio show` to view)")
args = p.parse_args()

suffix = ("" if args.target == "heart" else f"_{args.target}") + args.tag
RUN = (args.run or f"{args.graph}{suffix or '_base'}").replace(" ", "-")
os.makedirs("runs", exist_ok=True)
os.makedirs("docs/media", exist_ok=True)
CKPT = (f"runs/checkpoint{suffix or '_base'}--{RUN}.pt" if args.run
        else f"runs/checkpoint{suffix}.pt")
GIF, TARGET_PNG = f"docs/media/growth{suffix}.gif", f"docs/media/target{suffix}.png"

device = "mps" if torch.backends.mps.is_available() else "cpu"
torch.manual_seed(0)

# --animate: checkpoint is the source of truth (pos, edges, seed, weights).
# Do not rebuild the graph — --nodes / missing .npz would desync a cloud run.
track = None
if not args.animate:
    if args.track:
        import trackio as track
        track.init(project="gnca", name=RUN, config=vars(args))

    # --------------------------------------------------- graph + target ----
    is_cloud = args.target in POINTCLOUDS
    if is_cloud:
        # every node sits on the mesh; colour is baked from normals
        pos, target_np, seed_at = load_cloud(args.target, n_nodes=args.nodes)
        seed_at = np.asarray(seed_at, dtype=np.float32)
        dim = pos.shape[1]
        edges = knn_graph(pos, k=args.k)
        graph_kind = "cloud"
    elif args.graph == "ws" or (args.graph == "auto" and args.target == "ring"):
        target_fn, seed_pt = TARGETS[args.target]
        seed_at = np.array(seed_pt)
        dim = len(seed_at)
        pos, edges = watts_strogatz_graph(args.nodes, k=args.k, beta=args.beta)
        target_np = target_fn(pos)
        graph_kind = "ws"
    else:
        if args.target not in TARGETS:
            raise SystemExit(f"unknown target {args.target!r}")
        target_fn, seed_pt = TARGETS[args.target]
        seed_at = np.array(seed_pt)
        dim = len(seed_at)                       # the target decides 2-d or 3-d
        pos, edges = random_geometric_graph(args.nodes, k=args.k, dim=dim)
        target_np = target_fn(pos)
        graph_kind = "rgg"

    N = pos.shape[0]
    print(f"{args.target}: {dim}-d {graph_kind} graph, {N} nodes, "
          f"{int((target_np[:, 3] > 0.5).sum())} in the pattern")
    target = torch.from_numpy(target_np).to(device)                  # (N, 4)
    center = int(np.argmin(((pos - seed_at) ** 2).sum(1)))           # seed node

    # batched edges: replicate the graph args.batch times with node-index offsets
    offsets = torch.arange(args.batch, dtype=torch.int64)[:, None] * N
    bedges = (edges[None] + offsets[..., None]).permute(1, 0, 2).reshape(2, -1).to(device)
    edges = edges.to(device)

    model = GraphNCA(channels=args.channels).to(device)
    if args.init_from:
        sd = torch.load(args.init_from, weights_only=False)["model"]
        pad = load_rule(model, sd)
        print(f"warm-started from {args.init_from} (padded {pad} input cols)")
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # ------------------------------------------------------------ pool -----
    pool = seed_state(args.pool, N, args.channels, device, center)

    # ---------------------------------------------------------- damage -----
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

    def edge_energy(x):
        """Dirichlet energy: mean over edges of ||x_src - x_dst||^2. How much
        spatial structure the state carries; collapse during a rollout means
        perception is going blind (over-smoothing, issue #20)."""
        s, d = edges
        return ((x[s] - x[d]) ** 2).sum(-1).mean().item()

    def heal_probe(grow=80, heal=160):
        """Grow, punch a hole, heal. This is the number the whole recipe chases:
        training loss can fall while regeneration stays broken. Also samples
        Dirichlet energy along the growth rollout, the ablation-ladder metric."""
        global PROBE_BALL
        if PROBE_BALL is None:
            PROBE_BALL = torch.from_numpy(dmg.ball(
                pos, pattern, frac=0.25, center=int(pattern[len(pattern) // 2]))).to(device)
        with torch.no_grad():
            x = seed_state(1, N, args.channels, device, center)[0]
            energy = {}
            for t in range(1, grow + 1):
                x = model(x, edges) * alive_mask(x, edges, N)
                if t in (10, 20, 40, 80):
                    energy[f"dirich_t{t}"] = edge_energy(x)
            grown = ((x[:, :4] - target) ** 2).mean().item()
            x[PROBE_BALL] = 0.0
            for _ in range(heal):
                x = model(x, edges) * alive_mask(x, edges, N)
            healed = ((x[:, :4] - target) ** 2).mean().item()
            energy["dirich_healed"] = edge_energy(x)
            return grown, healed, energy

    def run_ca(x0, n_steps):
        """Roll out the CA, with pre-step alive masking like the growing NCA."""
        x = x0
        for _ in range(n_steps):
            m = alive_mask(x, bedges if x.shape[0] == args.batch * N else edges, N)
            x = model(x, bedges if x.shape[0] == args.batch * N else edges)
            x = x * m
        return x

    # ------------------------------------------------------------ train ----
    last_t = [time.time()]
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
            now = time.time()
            sps = 200 / max(now - last_t[0], 1e-9)
            last_t[0] = now
            print(f"step {step:6d}  loss {loss.item():.6f}  mse {mse.item():.6f}  {sps:.1f} steps/s")
            if track:
                track.log({"steps_per_sec": sps}, step=step)
        if step % 1000 == 0:
            grown, healed, energy = heal_probe()
            e_str = "  ".join(f"{k.removeprefix('dirich_')} {v:.4f}" for k, v in energy.items())
            print(f"    probe: grown {grown:.4f}  healed {healed:.4f}  |E| {e_str}")
            if track:
                track.log({"probe_grown": grown, "probe_healed": healed, **energy}, step=step)
        if step % 2000 == 0:
            torch.save({"model": model.state_dict(), "pos": pos, "target": target.cpu(),
                        "edges": edges.cpu(), "channels": args.channels, "run": RUN,
                        "center": center, "target_name": args.target, "dim": dim}, CKPT)

    torch.save({"model": model.state_dict(), "pos": pos, "target": target.cpu(),
                "edges": edges.cpu(), "channels": args.channels, "run": RUN,
                "center": center, "target_name": args.target, "dim": dim}, CKPT)
    print(f"saved {CKPT}")

# ------------------------------------------------------------ visualize ----
# Always reload the checkpoint so animate and post-train gifs share one path.
if not os.path.isfile(CKPT):
    raise SystemExit(f"missing checkpoint {CKPT}")

os.environ.pop("MPLBACKEND", None)  # don't inherit a notebook backend
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ckpt = torch.load(CKPT, weights_only=False)
pos = np.asarray(ckpt["pos"], dtype=np.float32)
tgt = ckpt["target"]
tgt = tgt.numpy() if torch.is_tensor(tgt) else np.asarray(tgt, dtype=np.float32)
edges = ckpt["edges"]
edges = edges.to(device) if torch.is_tensor(edges) else torch.as_tensor(edges, device=device)
N = pos.shape[0]
dim = int(ckpt.get("dim", pos.shape[1]))
channels = int(ckpt["channels"])
if "center" in ckpt:
    center = int(ckpt["center"])
else:
    center = int(np.argmin(((pos - 0.5) ** 2).sum(1)))

model = GraphNCA(channels=channels).to(device)
load_rule(model, ckpt["model"])
print(f"animate {CKPT}: {dim}-d, {N} nodes, seed={center}")


def draw(ax, rgba, title, spin=0.0):
    """Scatter the nodes. In 3-d, drop the dead ones and spin slowly, since a
    still frame of a shell reads as a disc."""
    ax.clear()
    a = np.clip(rgba[:, 3], 0, 1)
    c = np.clip(rgba[:, :3], 0, 1)
    if dim == 3:
        live = a > 0.1
        ax.set_facecolor("black")
        ax.scatter(pos[live, 0], pos[live, 1], pos[live, 2],
                   c=c[live], s=22, alpha=0.9, linewidths=0, depthshade=False)
        ax.set(xlim=(0.08, 0.92), ylim=(0.08, 0.92), zlim=(0.08, 0.92))
        ax.set_title(title, color="white")
        ax.view_init(elev=18, azim=spin)
        ax.set_box_aspect((1, 1, 1))
    else:
        ax.scatter(pos[:, 0], pos[:, 1], c=c, s=14, alpha=a, linewidths=0)
        ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal", title=title)
    ax.axis("off")

fig = plt.figure(figsize=(5, 5), dpi=120,
                 facecolor="black" if dim == 3 else "white")
ax = fig.add_subplot(projection="3d" if dim == 3 else None)
x = seed_state(1, N, model.channels, device, center)[0]
frames = []
with torch.no_grad():
    for t in range(120):
        frames.append(x[:, :4].cpu().numpy().copy())
        m = alive_mask(x, edges, N)
        x = model(x, edges) * m

anim = FuncAnimation(fig, lambda i: draw(ax, frames[i], f"graph NCA, t={i}", spin=-60 + i * 1.5),
                     frames=len(frames), interval=60)
anim.save(GIF, writer=PillowWriter(fps=16))
draw(ax, tgt, "target", spin=30)
plt.savefig(TARGET_PNG)
print(f"saved {GIF} and {TARGET_PNG}")

if track:  # post-train: attach the rollout + target to the run, then close it
    track.log({"growth_rollout": track.Video(GIF, caption=f"{RUN} growth", fps=16),
               "target": track.Image(TARGET_PNG, caption=f"{RUN} target")}, step=args.steps)
    track.finish()
