"""Download classic free meshes, sample surface point clouds, save .npz files.

    uv run python scripts/fetch_pointclouds.py
    uv run python scripts/fetch_pointclouds.py --nodes 2048 --only bunny spot

Writes data/pointclouds/{name}.npz with arrays:
    pos      (N, 3) float32   original mesh coords (training normalises them)
    normals  (N, 3) float32   area-weighted face normals at sample points
    name     scalar str

Raw OBJs cache under data/meshes/ (gitignored). The small .npz files are what
training and the web export load.
"""
from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MESH_DIR = ROOT / "data" / "meshes"
CLOUD_DIR = ROOT / "data" / "pointclouds"

# Small, recognisable meshes from Alec Jacobson's common-3d-test-models mirror.
# Each is a well-known scan/model; we only keep a few thousand surface samples.
MESHES = {
    "bunny": {
        "url": ("https://raw.githubusercontent.com/alecjacobson/"
                "common-3d-test-models/master/data/stanford-bunny.obj"),
        "file": "stanford-bunny.obj",
    },
    "spot": {
        "url": ("https://raw.githubusercontent.com/alecjacobson/"
                "common-3d-test-models/master/data/spot.obj"),
        "file": "spot.obj",
    },
    "armadillo": {
        "url": ("https://raw.githubusercontent.com/alecjacobson/"
                "common-3d-test-models/master/data/armadillo.obj"),
        "file": "armadillo.obj",
    },
    "teapot": {
        # Utah teapot, also in the common-3d-test-models set
        "url": ("https://raw.githubusercontent.com/alecjacobson/"
                "common-3d-test-models/master/data/teapot.obj"),
        "file": "teapot.obj",
    },
}


def download(url: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached {dest.relative_to(ROOT)}")
        return dest
    print(f"  downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "gnca-fetch/1.0"})
    with urllib.request.urlopen(req, timeout=120) as r, open(dest, "wb") as f:
        f.write(r.read())
    print(f"  wrote {dest.relative_to(ROOT)} ({dest.stat().st_size // 1024} KB)")
    return dest


def load_obj(path: Path):
    """Minimal OBJ loader: triangle faces only (fans quads). Returns verts, faces."""
    verts, faces = [], []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
            elif line.startswith("f "):
                # f v / f v/t / f v/t/n / f v//n  — 1-based indices, may be negative
                idx = []
                for tok in line.split()[1:]:
                    v = int(tok.split("/")[0])
                    idx.append(v - 1 if v > 0 else v)  # negative: relative
                # fan triangulation
                for i in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[i], idx[i + 1]))
    if not verts or not faces:
        raise ValueError(f"no triangle mesh in {path}")
    V = np.asarray(verts, dtype=np.float64)
    # resolve any relative indices
    n = len(V)
    F = np.asarray(faces, dtype=np.int64)
    F = np.where(F < 0, F + n, F)
    return V, F


def sample_surface(verts, faces, n_samples, rng):
    """Area-weighted surface samples + face normals at those points."""
    tri = verts[faces]                                    # (F, 3, 3)
    e1 = tri[:, 1] - tri[:, 0]
    e2 = tri[:, 2] - tri[:, 0]
    cross = np.cross(e1, e2)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    total = areas.sum()
    if total <= 0:
        raise ValueError("mesh has zero surface area")
    probs = areas / total
    face_idx = rng.choice(len(faces), size=n_samples, p=probs)

    # barycentric coords (square-root method for uniform triangle sampling)
    r1 = np.sqrt(rng.random(n_samples))
    r2 = rng.random(n_samples)
    a, b, c = 1 - r1, r1 * (1 - r2), r1 * r2
    chosen = tri[face_idx]
    pos = (a[:, None] * chosen[:, 0]
           + b[:, None] * chosen[:, 1]
           + c[:, None] * chosen[:, 2])

    normals = cross[face_idx]
    lens = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.clip(lens, 1e-12, None)
    return pos.astype(np.float32), normals.astype(np.float32)


def build_cloud(name, n_samples, seed):
    spec = MESHES[name]
    obj_path = download(spec["url"], MESH_DIR / spec["file"])
    print(f"  parsing {obj_path.name}")
    verts, faces = load_obj(obj_path)
    print(f"  mesh: {len(verts)} verts, {len(faces)} tris")
    rng = np.random.default_rng(seed)
    pos, normals = sample_surface(verts, faces, n_samples, rng)
    return pos, normals


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--nodes", type=int, default=1536,
                   help="surface samples per cloud (default 1536)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", nargs="+", choices=list(MESHES),
                   help="build just these clouds")
    args = p.parse_args()

    names = args.only or list(MESHES)
    CLOUD_DIR.mkdir(parents=True, exist_ok=True)
    for name in names:
        print(f"[{name}]")
        try:
            pos, normals = build_cloud(name, args.nodes, args.seed)
        except Exception as e:
            print(f"  FAILED: {e}")
            continue
        out = CLOUD_DIR / f"{name}.npz"
        np.savez_compressed(out, pos=pos, normals=normals, name=np.array(name))
        print(f"  saved {out.relative_to(ROOT)}  "
              f"({len(pos)} pts, bbox "
              f"{pos.min(0).round(3).tolist()} .. {pos.max(0).round(3).tolist()})")
    print("done. Train with e.g.:")
    print("  uv run python -u scripts/train.py --target bunny --damage 3 --tag _pc")


if __name__ == "__main__":
    main()
