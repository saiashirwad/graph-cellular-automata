"""The graphs the CA lives on. Each returns node positions and an edge index."""
import numpy as np
import torch


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
