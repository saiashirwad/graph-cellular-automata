"""The patterns the CA grows, sampled at the graph's node positions.

Each target returns (N, 4) float32 RGBA, alpha 1 where the pattern is. TARGETS
also records where to put the seed node, which is not always the centre: the
annulus has nothing there.

Shape matters for a regeneration demo. A blob heals invisibly, since a filled
hole looks like any other filled hole. A shape with parts, arms, a hole, two
lobes, tells you whether it healed *correctly*, and colouring by angle means a
regrown arm comes back in a colour you can check.
"""
import numpy as np


def hue_rgb(h):
    """HSV to RGB at full saturation and value. h in [0, 1], shape (N,)."""
    i = (h * 6).astype(int) % 6
    f = h * 6 - np.floor(h * 6)
    q, t = 1 - f, f
    one, zero = np.ones_like(h), np.zeros_like(h)
    palette = np.stack([  # (N, 6 hues, 3 channels)
        np.stack([one, t, zero], -1), np.stack([q, one, zero], -1),
        np.stack([zero, one, t], -1), np.stack([zero, q, one], -1),
        np.stack([t, zero, one], -1), np.stack([one, zero, q], -1),
    ], axis=1)
    return np.take_along_axis(palette, i[:, None, None], axis=1)[:, 0, :]


def _rgba(rgb, inside):
    rgba = np.concatenate([rgb, np.ones((len(rgb), 1), np.float32)], axis=-1)
    rgba[~inside] = 0.0
    return rgba.astype(np.float32)


def _polar(pos, cx=0.5, cy=0.5):
    dx, dy = pos[:, 0] - cx, pos[:, 1] - cy
    r = np.sqrt(dx * dx + dy * dy)
    h = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)      # 0..1 by angle
    return r, h


def heart_target(pos):
    """A rainbow-ringed heart. The original, and a solid blob."""
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
    return _rgba(rgb, inside)


def star_target(pos, arms=5):
    """A five-armed starfish, each arm its own hue.

    The best shape here for watching regeneration. Cut an arm off and the
    question is not just whether tissue comes back, but whether it comes back
    as an arm, in the right place, in the right colour.
    """
    r, h = _polar(pos)
    inside = r < 0.20 + 0.13 * np.cos(arms * (h * 2 * np.pi))
    return _rgba(hue_rgb(h), inside)


def annulus_target(pos):
    """A ring with a hole, coloured by angle.

    A blob can heal by filling in. This one cannot: closing the hole is as
    wrong as leaving a gap, so the rule has to rebuild a topology, not an area.
    """
    r, h = _polar(pos)
    inside = (r > 0.17) & (r < 0.33)
    return _rgba(hue_rgb(h), inside)


def lobes_target(pos):
    """Two discs joined by a thin bridge.

    Pairs with the edge-cutting demo. Sever the bridge and the halves are on
    their own, each holding its own colour.
    """
    left = ((pos[:, 0] - 0.31) ** 2 + (pos[:, 1] - 0.5) ** 2) < 0.031
    right = ((pos[:, 0] - 0.69) ** 2 + (pos[:, 1] - 0.5) ** 2) < 0.031
    bridge = (np.abs(pos[:, 1] - 0.5) < 0.035) & (np.abs(pos[:, 0] - 0.5) < 0.21)
    inside = left | right | bridge

    rgb = np.zeros((len(pos), 3), np.float32)
    rgb[:] = [0.30, 0.78, 0.95]                            # left lobe, cyan
    rgb[right] = [1.00, 0.42, 0.60]                        # right lobe, rose
    rgb[bridge & ~left & ~right] = [0.95, 0.85, 0.35]      # the bridge, amber
    return _rgba(rgb, inside)


def ring_target(pos):
    """Rainbow-by-angle on the ring graph; alpha=1 everywhere.

    Pairs with watts_strogatz_graph: the CA must grow a full color wheel
    around the ring from one seed, jumping through shortcut edges.
    """
    _, h = _polar(pos)
    rgba = np.concatenate([hue_rgb(h), np.ones((len(pos), 1), np.float32)], axis=-1)
    return rgba.astype(np.float32)


# ------------------------------------------------------------------ 3-d ----
# A graph has no reason to be flat. These live in [0,1]^3 and pair with
# random_geometric_graph(dim=3). They paint a thin shell inside a random cube,
# so only a fraction of nodes are live — fine for experiments, thin for demos.
# Prefer the surface point clouds in gnca.pointclouds (bunny, spot, …) when you
# want every node on the shape. Damage in 3-d is genuinely harder: a wound has
# a whole extra direction to heal from, and you can hide it behind the shape.

def sphere_target(pos):
    """A hollow shell, coloured by direction. Solid inside is empty."""
    d = pos - 0.5
    r = np.sqrt((d ** 2).sum(1))
    inside = (r > 0.19) & (r < 0.39)
    h = (np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi)
    rgb = hue_rgb(h) * (0.45 + 0.55 * (d[:, 2] / 0.39 + 1) / 2)[:, None]
    return _rgba(rgb.astype(np.float32), inside)


def torus_target(pos, R=0.27, r_tube=0.19):
    """A donut. The hole is the point: healing cannot just fill volume."""
    d = pos - 0.5
    q = np.sqrt(d[:, 0] ** 2 + d[:, 1] ** 2) - R
    inside = (q ** 2 + d[:, 2] ** 2) < r_tube ** 2
    h = (np.arctan2(d[:, 1], d[:, 0]) + np.pi) / (2 * np.pi)   # hue around the ring
    return _rgba(hue_rgb(h), inside)


def jack_target(pos, half=0.42, thick=0.17):
    """Three arms along x, y and z, each its own colour.

    The 3-d answer to the starfish. Lop off an arm and you can see at a glance
    whether the right one grew back.
    """
    d = pos - 0.5
    ax = [(np.abs(d[:, i]) < half) &
          (np.sqrt((d[:, [j for j in range(3) if j != i]] ** 2).sum(1)) < thick)
          for i in range(3)]
    inside = ax[0] | ax[1] | ax[2]

    rgb = np.zeros((len(pos), 3), np.float32)
    for a, c in zip(ax, ([1.00, 0.42, 0.60], [0.55, 0.92, 0.45], [0.35, 0.70, 1.00])):
        rgb[a] = c
    return _rgba(rgb, inside)


# name -> (function, where to put the seed node). The length of the seed point
# is what tells the rest of the code whether this target is 2-d or 3-d.
TARGETS = {
    "heart":   (heart_target,   (0.50, 0.45)),
    "star":    (star_target,    (0.50, 0.50)),
    "annulus": (annulus_target, (0.75, 0.50)),   # on the ring, not in the hole
    "lobes":   (lobes_target,   (0.31, 0.50)),   # in the left lobe
    "ring":    (ring_target,    (1.00, 0.50)),   # node 0 on the WS ring
    "sphere":  (sphere_target,  (0.80, 0.50, 0.50)),
    "torus":   (torus_target,   (0.77, 0.50, 0.50)),
    "jack":    (jack_target,    (0.50, 0.50, 0.50)),
}
