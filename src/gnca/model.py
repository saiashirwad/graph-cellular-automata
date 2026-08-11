"""The update rule: one shared MLP applied to every node, Distill-NCA style."""
import torch
import torch.nn.functional as F
from torch import nn


class GraphNCA(nn.Module):
    """One shared update rule applied to every node, Distill-NCA style.

    Perception (message passing): each node sees
        [ own state, mean of neighbor states, mean of (neighbor - own), log(1 + deg) ]
    The degree scalar restores the neighbor COUNT that mean aggregation erases.
    The difference term is gated per edge and channel (Perona-Malik style):
    the rule can turn diffusion OFF across pattern boundaries instead of
    smearing them (issue #18).
    Update: tiny MLP -> residual increment, applied stochastically.
    """

    def __init__(self, channels=16, hidden=128):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(3 * channels + 1, hidden), nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
        )
        # zero-init the last layer: initial rule is the identity
        nn.init.zeros_(self.net[-1].weight)
        # per-channel edge conductivity. Zero-init => 2*sigmoid(0) = 1 exactly,
        # so at init the gate is identity diffusion and warm starts are exact.
        self.gate = nn.Linear(3 * channels, channels)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x, edge_index, update_rate=0.5, deg=None):
        """x: (B*N, C) node states for a batch of B copies of the graph.

        deg: optional precomputed (B*N, 1) in-degree. The graph never changes
        during a run, so the training loop computes it once and passes it in;
        computed here when not given."""
        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]
        if deg is None:
            deg = torch.zeros(n, 1, device=x.device, dtype=x.dtype)
            ones = torch.ones(src.shape[0], 1, device=x.device, dtype=x.dtype)
            deg.index_add_(0, dst, ones)
        inv_deg = 1.0 / deg.clamp(min=1)
        mean_n = torch.zeros_like(x).index_add_(0, dst, x[src]) * inv_deg

        # The gate is linear in [x_src, x_dst, |x_src - x_dst|], so the two
        # endpoint terms are computed once on N nodes and gathered per edge,
        # instead of one (E, 3C) matmul over edges (the #18 slowdown).
        d = x[src] - x[dst]
        w1, w2, w3 = self.gate.weight.split(self.channels, dim=1)
        a = F.linear(x, torch.cat([w1, w2], dim=0))          # (N, 2C)
        g = 2 * torch.sigmoid(a[:, :self.channels][src] + a[:, self.channels:][dst]
                              + F.linear(d.abs(), w3, self.gate.bias))
        mean_diff = torch.zeros_like(x).index_add_(0, dst, g * d) * inv_deg

        # log1p(deg) LAST: neighbor count (issue #17). Appended last so
        # warm-start zero-padding of the first layer stays a column pad.
        z = torch.cat([x, mean_n, mean_diff, torch.log1p(deg)], dim=-1)
        dx = self.net(z)

        # stochastic per-node update (classic NCA trick for async robustness)
        mask = (torch.rand(n, 1, device=x.device) < update_rate).to(x.dtype)
        return x + dx * mask


def load_rule(model, sd):
    """Load a rule state dict, zero-padding the first layer's input columns if
    the checkpoint's percept is narrower (pre-#17 checkpoints lack the degree
    feature) and leaving any missing gate at its zero (identity) init.
    Zero columns + identity gate => the loaded rule is functionally identical."""
    w = "net.0.weight"
    pad = model.net[0].weight.shape[1] - sd[w].shape[1]
    if pad > 0:
        zpad = torch.zeros(sd[w].shape[0], pad, device=sd[w].device)
        sd[w] = torch.cat([sd[w], zpad], dim=1)
    missing = model.load_state_dict(sd, strict=False).missing_keys
    assert all(k.startswith("gate.") for k in missing), f"unexpected missing keys: {missing}"
    return pad


def alive_mask(x, edge_index, threshold=0.1):
    """Node lives if it or any neighbor has alpha > threshold (graph 3x3 mask)."""
    alpha = x[:, 3:4]
    neigh_max = torch.full_like(alpha, -1e9)
    neigh_max.index_reduce_(0, edge_index[1], alpha[edge_index[0]], "amax", include_self=True)
    return (torch.maximum(alpha, neigh_max) > threshold).to(x.dtype)


def seed_state(batch, n_nodes, channels, device, center=0):
    """All-empty states except one 'seed' node (usually nearest pattern center)."""
    x = torch.zeros(batch, n_nodes, channels, device=device)
    x[:, center, 3:] = 1.0
    return x
