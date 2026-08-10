"""The patterns the CA grows, sampled at the graph's node positions."""
import numpy as np


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
