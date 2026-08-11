"""Load a trained checkpoint back into a runnable model, and roll it out.

Everything that is not training (eval, render, export, animate) needs the same
handful of things: the model with its weights loaded tolerantly, the graph,
the seed node, the target. Each script used to redo this by hand, and the ones
using a strict load_state_dict crashed on any checkpoint older than the
current percept. One loader, one rollout.
"""
from dataclasses import dataclass

import numpy as np
import torch

from gnca.model import GraphNCA, alive_mask, load_rule
from gnca.targets import TARGETS


@dataclass
class Checkpoint:
    model: GraphNCA        # weights loaded, on device
    pos: np.ndarray        # (N, dim) float32
    edges: torch.Tensor    # (2, E) long, on device
    target: np.ndarray     # (N, 4) float32
    center: int            # seed node
    channels: int
    dim: int
    pad: int               # input columns zero-padded into the percept (old ckpts)

    @property
    def n(self):
        return self.pos.shape[0]

    @property
    def edges_np(self):
        return self.edges.cpu().numpy()


def load_checkpoint(path, device=None):
    device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    ckpt = torch.load(path, weights_only=False, map_location="cpu")

    pos = np.asarray(ckpt["pos"], dtype=np.float32)
    e = ckpt["edges"]
    e = e.numpy() if torch.is_tensor(e) else np.asarray(e)
    edges = torch.from_numpy(e.astype(np.int64)).to(device)
    t = ckpt["target"]
    target = (t.numpy() if torch.is_tensor(t) else np.asarray(t)).astype(np.float32)

    if "center" in ckpt:
        center = int(ckpt["center"])
    else:
        # checkpoints old enough to lack a seed node are all the heart; its
        # seed hint is off-centre, so look it up rather than guessing 0.5
        hint = TARGETS.get(ckpt.get("target_name") or "heart", TARGETS["heart"])[1]
        center = int(np.argmin(((pos - np.asarray(hint, dtype=np.float32)) ** 2).sum(1)))

    channels = int(ckpt["channels"])
    model = GraphNCA(channels=channels).to(device)
    pad = load_rule(model, ckpt["model"])
    dim = int(ckpt.get("dim", pos.shape[1]))
    return Checkpoint(model, pos, edges, target, center, channels, dim, pad)


@torch.no_grad()
def rollout(model, x, edges, n_steps):
    """n_steps of grow: pre-step alive mask, update, apply mask.

    Training keeps its own loop in scripts/train.py, which needs grad."""
    for _ in range(n_steps):
        x = model(x, edges) * alive_mask(x, edges)
    return x
