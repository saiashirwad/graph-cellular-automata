"""Score a checkpoint on four kinds of damage: grow, break, watch it heal.

    uv run python eval_damage.py checkpoint.pt checkpoint_dmg.pt

Prints, for each damage type, the alive% and RGBA MSE at the end of growth,
right after the hit, and through healing. One row per checkpoint so a
damage-trained rule can be compared against the plain one.
"""
import argparse
import numpy as np
import torch

from gnca import GraphNCA, alive_mask, seed_state

p = argparse.ArgumentParser()
p.add_argument("ckpts", nargs="*", default=["checkpoint.pt"])
p.add_argument("--grow", type=int, default=80)
p.add_argument("--heal", type=int, default=160)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
rng = np.random.default_rng(args.seed)


def damage_sets(pos, target_np):
    """Node index arrays for each damage type, plus a printable name."""
    pattern = np.flatnonzero(target_np[:, 3] > 0.5)
    ppos = pos[pattern]
    cx, cy = ppos.mean(0)

    def ball(frac):
        c = pattern[rng.integers(len(pattern))]
        d2 = ((pos - pos[c]) ** 2).sum(1)
        r2 = np.quantile(d2[pattern], frac)
        return np.flatnonzero(d2 <= r2)

    band = np.flatnonzero((pos[:, 1] > cy - 0.08) & (pos[:, 1] < cy + 0.08))
    half = pattern[ppos[:, 0] > cx]
    scatter = rng.choice(pattern, size=int(0.25 * len(pattern)), replace=False)
    return [
        ("band (horizontal slice)", band),
        ("ball (25% of pattern)", ball(0.25)),
        ("random 25% of nodes", scatter),
        ("right half", half),
    ]


def run(ckpt_path):
    # reseed per checkpoint: every model must face the same damage regions and
    # the same stochastic update draws, or the comparison measures noise
    global rng
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)

    ckpt = torch.load(ckpt_path, weights_only=False)
    pos, edges = ckpt["pos"], ckpt["edges"].to(device)
    C, N = ckpt["channels"], pos.shape[0]
    target = ckpt["target"]
    target = target if torch.is_tensor(target) else torch.from_numpy(target)
    target_np = target.numpy()
    target = target.to(device)
    model = GraphNCA(channels=C).to(device)
    model.load_state_dict(ckpt["model"])
    center = int(np.argmin(((pos - [0.5, 0.45]) ** 2).sum(1)))
    live_target = float((target_np[:, 3] > 0.5).mean())

    def stats(x):
        alive = (x[:, 3] > 0.1).float().mean().item()
        mse = ((x[:, :4] - target) ** 2).mean().item()
        return alive / max(live_target, 1e-6) * 100, mse

    print(f"\n=== {ckpt_path} ===")
    with torch.no_grad():
        # one shared growth rollout, then branch per damage type
        x = seed_state(1, N, C, device, center)[0]
        for _ in range(args.grow):
            x = model(x, edges) * alive_mask(x, edges, N)
        grown = x.clone()
        a, m = stats(grown)
        print(f"  grown (t={args.grow}):  alive {a:5.1f}%  mse {m:.4f}")

        for name, nodes in damage_sets(pos, target_np):
            x = grown.clone()
            x[torch.from_numpy(nodes).to(device)] = 0.0
            a0, m0 = stats(x)
            curve = []
            for t in range(1, args.heal + 1):
                x = model(x, edges) * alive_mask(x, edges, N)
                if t % 40 == 0:
                    curve.append((t,) + stats(x))
            tail = "  ".join(f"t+{t}: {a:5.1f}%/{m:.4f}" for t, a, m in curve)
            print(f"  {name:24s} hit: {a0:5.1f}%/{m0:.4f}  ->  {tail}")


for c in args.ckpts:
    run(c)
