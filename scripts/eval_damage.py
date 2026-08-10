"""Score a checkpoint on four kinds of damage: grow, break, watch it heal.

    uv run python scripts/eval_damage.py runs/checkpoint.pt runs/checkpoint_dmg.pt

Prints, for each damage type, the alive% and RGBA MSE at the end of growth,
right after the hit, and through healing. One row per checkpoint so a
damage-trained rule can be compared against the plain one.
"""
import argparse

import numpy as np
import torch

from gnca import GraphNCA, alive_mask, seed_state
from gnca import damage as dmg

p = argparse.ArgumentParser()
p.add_argument("ckpts", nargs="*", default=["runs/checkpoint.pt"])
p.add_argument("--grow", type=int, default=80)
p.add_argument("--heal", type=int, default=160)
p.add_argument("--seed", type=int, default=0)
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"


def damage_sets(pos, target_np):
    """Node index arrays for each damage type, plus a printable name.

    Each random shape gets its own generator seeded from its position in the
    list, so adding or reordering a damage type cannot silently move the
    others. A shared generator did exactly that once already.
    """
    pattern = dmg.pattern_nodes(target_np)
    seeded = lambda k: np.random.default_rng([args.seed, k])
    return [
        ("band (horizontal slice)", dmg.band(pos, pattern)),
        ("ball (25% of pattern)", dmg.ball(pos, pattern, frac=0.25, rng=seeded(1))),
        ("random 25% of nodes", dmg.scatter(pattern, frac=0.25, rng=seeded(2))),
        ("right half", dmg.half(pos, pattern)),
    ]


def run(ckpt_path):
    # reseed per checkpoint: every model must face the same stochastic update
    # draws, or the comparison measures noise. Damage regions are seeded
    # independently in damage_sets.
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
