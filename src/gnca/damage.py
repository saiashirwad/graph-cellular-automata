"""Ways to break a grown pattern, shared by training, evaluation and rendering.

These used to live in four scripts with four slightly different definitions,
which meant the eval was only measuring what training optimized by luck. Every
function here returns node indices to zero, or an edge keep-mask, and nothing
else decides what "a ball" or "a band" means.
"""
import numpy as np


def pattern_nodes(target, threshold=0.5):
    """Indices of the nodes the target actually paints.

    Damage outside the pattern is a no-op the rule already handles, so every
    shape below is defined relative to these nodes.
    """
    alpha = np.asarray(target)[:, 3]
    return np.flatnonzero(alpha > threshold)


def ball(pos, pattern, frac=0.25, center=None, rng=None):
    """A blob around one node, covering `frac` of the pattern.

    The radius is a quantile of the squared distances to pattern nodes, so
    `frac` means the same thing regardless of how the nodes are spread out.
    """
    rng = rng or np.random.default_rng()
    if center is None:
        center = pattern[rng.integers(len(pattern))]
    d2 = ((pos - pos[center]) ** 2).sum(1)
    return np.flatnonzero(d2 <= np.quantile(d2[pattern], frac))


def scatter(pattern, frac=0.25, rng=None):
    """A random fraction of the pattern nodes. Damage with no shape at all."""
    rng = rng or np.random.default_rng()
    return rng.choice(pattern, size=max(1, int(frac * len(pattern))), replace=False)


def band(pos, pattern, height=0.16):
    """A horizontal slice through the middle of the pattern."""
    cy = pos[pattern, 1].mean()
    return np.flatnonzero(np.abs(pos[:, 1] - cy) < height / 2)


def half(pos, pattern, axis=0):
    """Everything on the far side of the pattern's centre. axis 0 = right."""
    c = pos[pattern, axis].mean()
    return pattern[pos[pattern, axis] > c]


# --------------------------------------------------------------- edges -----
# Cutting edges instead of nodes: the graph-native damage a grid cannot have.

def cut_across(pos, edges, y):
    """Keep-mask dropping every edge that crosses the horizontal line y."""
    e = np.asarray(edges)
    return (pos[e[0], 1] < y) == (pos[e[1], 1] < y)


def cut_random(edges, frac, rng=None):
    """Keep-mask dropping a random fraction of all edges."""
    rng = rng or np.random.default_rng()
    return rng.random(np.asarray(edges).shape[1]) >= frac
