"""
Graph Neural Cellular Automata (GNCA) - core pieces.

A Distill.pub-style "growing" neural CA, but the grid is replaced by an
arbitrary graph. Each node is a cell with a state vector; edges define the
neighborhood. Perception = GNN-style message passing instead of Sobel convs.

Pure PyTorch, no torch_geometric needed. Runs on CPU or Apple Silicon (MPS).
"""
import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------- graph ----

def random_geometric_graph(n_nodes=1024, k=8, seed=0):
    """Uniform random points in the unit square, edges = k-nearest neighbors.

    Returns:
        pos:        (N, 2) float32 array of node positions in [0, 1]^2
        edge_index: (2, E) long tensor, edges in both directions
    """
    rng = np.random.default_rng(seed)
    pos = rng.random((n_nodes, 2), dtype=np.float32)

    # k-NN via pairwise distances (fine for a few thousand nodes)
    d2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn_idx = np.argpartition(d2, k, axis=1)[:, :k]          # (N, k)

    src = np.repeat(np.arange(n_nodes), k)
    dst = nn_idx.reshape(-1)
    edges = np.stack([src, dst])
    edges = np.concatenate([edges, edges[::-1]], axis=1)    # bidirectional
    edges = np.unique(edges, axis=1)                        # dedupe
    return pos, torch.from_numpy(edges.astype(np.int64))


def watts_strogatz_graph(n_nodes=1024, k=8, beta=0.05, seed=0):
    """Watts-Strogatz small-world graph: ring lattice + rewired shortcuts.

    Nodes live on a circle (positions = the ring itself). Growth spreads
    locally around the ring and occasionally jumps through a shortcut edge.

    Returns:
        pos:        (N, 2) float32 positions on the unit circle, mapped to [0,1]^2
        edge_index: (2, E) long tensor, edges in both directions
    """
    rng = np.random.default_rng(seed)
    assert k % 2 == 0
    edges = set()
    for i in range(n_nodes):
        for d in range(1, k // 2 + 1):
            j = (i + d) % n_nodes
            # rewire this edge with probability beta
            if rng.random() < beta:
                candidates = np.setdiff1d(np.arange(n_nodes), [i])
                j = int(rng.choice(candidates))
            edges.add((min(i, j), max(i, j)))
    e = np.array(sorted(edges)).T
    e = np.concatenate([e, e[::-1]], axis=1)                # bidirectional

    theta = np.arange(n_nodes) / n_nodes * 2 * np.pi
    pos = np.stack([np.cos(theta), np.sin(theta)], axis=1).astype(np.float32)
    pos = (pos + 1.0) / 2.0                                 # -> [0, 1]^2
    return pos, torch.from_numpy(e.astype(np.int64))


# --------------------------------------------------------------- target ----

def heart_target(pos):
    """Target pattern: a rainbow-ringed heart. The graph 'grows' this.

    Returns (N, 4) float32: rgb + alpha (alpha=1 inside the heart).
    """
    x = (pos[:, 0] - 0.5) * 2.6
    y = (pos[:, 1] - 0.45) * 2.6
    a = x * x + y * y - 1.0
    inside = (a * a * a - x * x * y * y * y) < 0            # heart inequality

    r = np.sqrt(x * x + y * y)
    rgb = np.stack([
        0.5 + 0.5 * np.sin(8 * r),
        0.5 + 0.5 * np.sin(8 * r + 2.1),
        0.5 + 0.5 * np.sin(8 * r + 4.2),
    ], axis=-1)
    rgba = np.concatenate([rgb, np.ones_like(r[..., None])], axis=-1)
    rgba[~inside] = 0.0                                     # outside: empty
    return rgba.astype(np.float32)


def ring_target(pos):
    """Rainbow-by-angle on the ring graph; alpha=1 everywhere.

    Pairs with watts_strogatz_graph: the CA must grow a full color wheel
    around the ring from one seed, jumping through shortcut edges.
    """
    theta = np.arctan2(pos[:, 1] - 0.5, pos[:, 0] - 0.5)    # -pi..pi
    h = (theta + np.pi) / (2 * np.pi)                       # 0..1 hue
    # cheap HSV->RGB with s=v=1
    i = (h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    q, t = 1 - f, f
    one, zero = np.ones_like(h), np.zeros_like(h)
    palette = np.stack([  # (N, 6 hues, 3 channels)
        np.stack([one, t, zero], -1), np.stack([q, one, zero], -1),
        np.stack([zero, one, t], -1), np.stack([zero, q, one], -1),
        np.stack([t, zero, one], -1), np.stack([one, zero, q], -1),
    ], axis=1)
    rgb = np.take_along_axis(palette, i[:, None, None], axis=1)[:, 0, :]
    rgba = np.concatenate([rgb, np.ones((*h.shape, 1), np.float32)], axis=-1)
    return rgba.astype(np.float32)


# ---------------------------------------------------------------- model ----

class GraphNCA(nn.Module):
    """One shared update rule applied to every node, Distill-NCA style.

    Perception (message passing): each node sees
        [ own state, mean of neighbor states, mean of (neighbor - own) ]
    Update: tiny MLP -> residual increment, applied stochastically.
    """

    def __init__(self, channels=16, hidden=128):
        super().__init__()
        self.channels = channels
        self.net = nn.Sequential(
            nn.Linear(3 * channels, hidden), nn.ReLU(),
            nn.Linear(hidden, channels, bias=False),
        )
        # zero-init the last layer: initial rule is the identity
        nn.init.zeros_(self.net[-1].weight)

    def forward(self, x, edge_index, update_rate=0.5):
        """x: (B*N, C) node states for a batch of B copies of the graph."""
        src, dst = edge_index[0], edge_index[1]
        n = x.shape[0]
        mean_n = torch.zeros_like(x).index_add_(0, dst, x[src])
        deg = torch.zeros(n, 1, device=x.device, dtype=x.dtype)
        deg.index_add_(0, dst, torch.ones(src.shape[0], 1, device=x.device, dtype=x.dtype))
        mean_n = mean_n / deg.clamp(min=1)
        mean_diff = torch.zeros_like(x).index_add_(0, dst, x[src] - x[dst]) / deg.clamp(min=1)

        z = torch.cat([x, mean_n, mean_diff], dim=-1)
        dx = self.net(z)

        # stochastic per-node update (classic NCA trick for async robustness)
        mask = (torch.rand(n, 1, device=x.device) < update_rate).to(x.dtype)
        return x + dx * mask


def alive_mask(x, edge_index, n_nodes, threshold=0.1):
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
