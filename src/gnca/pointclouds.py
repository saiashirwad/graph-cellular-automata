"""Surface point clouds as 3-d growth targets.

The procedural sphere/torus/jack paint a thin shell inside a random cube of
nodes, so only ~15-20% of the graph is the pattern and demos look like dust.
Here every node sits on a real mesh surface: the graph is the shape, alpha is
1 everywhere, and colour comes from surface normals so a regrown ear is a
regrown colour you can check.

Clouds live as .npz under data/pointclouds/ (see scripts/fetch_pointclouds.py).
"""
from pathlib import Path

import numpy as np

from gnca.targets import hue_rgb

# repo root: src/gnca/pointclouds.py -> ../../
DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "pointclouds"

# name -> default seed hint in [0,1]^3 (node nearest this point becomes the seed).
# Hints bias toward a distinctive tip so growth has a clear starting place.
POINTCLOUDS = {
    "bunny":     (0.50, 0.85, 0.55),   # near the ears
    "spot":      (0.50, 0.75, 0.70),   # head / horns
    "armadillo": (0.50, 0.80, 0.55),   # head
    "teapot":    (0.75, 0.55, 0.55),   # spout tip
}


def cloud_path(name: str) -> Path:
    return DATA_DIR / f"{name}.npz"


def normalize_unit_cube(pos, margin=0.08):
    """Center and scale so the cloud sits in [margin, 1-margin]^d."""
    pos = np.asarray(pos, dtype=np.float32)
    lo, hi = pos.min(0), pos.max(0)
    span = float((hi - lo).max()) or 1.0
    mid = (lo + hi) / 2
    # map longest axis to [margin, 1-margin]
    scale = (1.0 - 2 * margin) / span
    return ((pos - mid) * scale + 0.5).astype(np.float32)


def color_by_normals(normals):
    """Rainbow by azimuth, dimmed by elevation — readable regeneration cue."""
    n = np.asarray(normals, dtype=np.float32)
    # guard zero normals
    lens = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.clip(lens, 1e-8, None)
    h = (np.arctan2(n[:, 1], n[:, 0]) + np.pi) / (2 * np.pi)
    elev = (n[:, 2] + 1) * 0.5                          # 0..1
    rgb = hue_rgb(h) * (0.45 + 0.55 * elev)[:, None]
    rgba = np.concatenate([rgb, np.ones((len(n), 1), np.float32)], axis=-1)
    return rgba.astype(np.float32)


def estimate_normals(pos, k=12):
    """PCA normals from k-NN. Orientation is flipped to point away from centroid."""
    pos = np.asarray(pos, dtype=np.float32)
    n = len(pos)
    k = min(k, n - 1)
    d2 = ((pos[:, None] - pos[None]) ** 2).sum(-1)
    np.fill_diagonal(d2, np.inf)
    nn = np.argpartition(d2, k, axis=1)[:, :k]
    normals = np.zeros_like(pos)
    for i in range(n):
        pts = pos[nn[i]] - pos[i]
        _, _, vt = np.linalg.svd(pts, full_matrices=False)
        normals[i] = vt[-1]
    # orient outward from centroid
    centroid = pos.mean(0)
    flip = ((normals * (pos - centroid)).sum(1) < 0)
    normals[flip] *= -1
    return normals.astype(np.float32)


def load_cloud(name, n_nodes=None, seed_hint=None):
    """Load a prepared cloud. Returns (pos, rgba, seed_xyz).

    n_nodes: if set and smaller than the stored cloud, subsample deterministically.
    seed_hint: override the default seed location in [0,1]^3.
    """
    if name not in POINTCLOUDS:
        raise KeyError(f"unknown point cloud {name!r}; have {sorted(POINTCLOUDS)}")
    path = cloud_path(name)
    if not path.exists():
        raise FileNotFoundError(
            f"missing {path}. Run: uv run python scripts/fetch_pointclouds.py"
        )
    data = np.load(path)
    pos = np.asarray(data["pos"], dtype=np.float32)
    if "normals" in data:
        normals = np.asarray(data["normals"], dtype=np.float32)
    else:
        normals = estimate_normals(pos)

    if n_nodes is not None and n_nodes < len(pos):
        rng = np.random.default_rng(0)
        idx = np.sort(rng.choice(len(pos), size=n_nodes, replace=False))
        pos, normals = pos[idx], normals[idx]
    elif n_nodes is not None and n_nodes > len(pos):
        # cannot invent surface points; use what we have
        pass

    pos = normalize_unit_cube(pos)
    rgba = color_by_normals(normals)
    hint = np.asarray(seed_hint if seed_hint is not None else POINTCLOUDS[name],
                      dtype=np.float32)
    return pos, rgba, hint


def list_available():
    """Names with an on-disk .npz ready to train."""
    return sorted(n for n in POINTCLOUDS if cloud_path(n).exists())
