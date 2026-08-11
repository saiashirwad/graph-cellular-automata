"""Visualize the 12 hidden channels as "morphogen fields" via PCA -> RGB.

    uv run python scripts/render/hidden_channels.py
Saves docs/media/hidden_channels.gif: a side-by-side of the grown RGBA pattern
and the PCA-projected hidden chemistry at each step.
"""
import argparse
import os

os.environ.pop("MPLBACKEND", None)
import numpy as np
import torch
from sklearn.decomposition import PCA

from gnca import alive_mask, load_checkpoint, seed_state

p = argparse.ArgumentParser()
p.add_argument("--ckpt", default="runs/checkpoint.pt")
p.add_argument("--out", default="docs/media/hidden_channels.gif")
p.add_argument("--weights", default="runs/pca_weights.npz",
               help="PCA basis, read by the web demo's hidden-channel view")
args = p.parse_args()

device = "mps" if torch.backends.mps.is_available() else "cpu"
ck = load_checkpoint(args.ckpt, device)
pos, edges, model = ck.pos, ck.edges, ck.model
C, N = ck.channels, ck.n

# ---- collect hidden states over a rollout to fit PCA ----
x = seed_state(1, N, C, device, ck.center)[0]
hidden_snapshots = []
with torch.no_grad():
    for t in range(120):
        hidden = x[:, 4:C].cpu().numpy()
        alive = hidden[hidden[:, 0] != 0]
        hidden_snapshots.append(alive if alive.shape[0] > 10 else hidden)
        m = alive_mask(x, edges)
        x = model(x, edges) * m

# fit PCA on all collected hidden states (only alive nodes)
all_hidden = np.vstack(hidden_snapshots)
pca = PCA(n_components=3)
pca.fit(all_hidden)
print(f"PCA explained variance: {pca.explained_variance_ratio_}, "
      f"cumulative: {pca.explained_variance_ratio_.sum():.3f}")

# ---- re-run rollout, recording PCA-projected frames ----
x = seed_state(1, N, C, device, ck.center)[0]
rgba_frames, pca_frames = [], []
with torch.no_grad():
    for t in range(120):
        rgba_frames.append(x[:, :4].cpu().numpy().copy())
        hidden = x[:, 4:C].cpu().numpy()
        pca_rgb = pca.transform(hidden)                # (N, 3)
        # normalize each component to [0, 1]
        for c in range(3):
            col = pca_rgb[:, c]
            rng = col.max() - col.min()
            if rng > 1e-8:
                pca_rgb[:, c] = (col - col.min()) / rng
        pca_frames.append(pca_rgb)
        m = alive_mask(x, edges)
        x = model(x, edges) * m

# ---- animate side by side ----
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), dpi=110)

def draw(i):
    for ax in (ax1, ax2):
        ax.clear(); ax.axis("off"); ax.set_aspect("equal")
    # left: grown pattern
    rgba = np.clip(rgba_frames[i], 0, 1)
    ax1.scatter(pos[:, 0], pos[:, 1], c=rgba[:, :3], s=14,
                alpha=rgba[:, 3], linewidths=0)
    ax1.set_title(f"grown pattern (t={i})")
    # right: PCA hidden channels
    pca_r = np.clip(pca_frames[i], 0, 1)
    alive_a = rgba_frames[i][:, 3]
    ax2.scatter(pos[:, 0], pos[:, 1], c=pca_r, s=14,
                alpha=np.clip(alive_a, 0.1, 1), linewidths=0)
    ax2.set_title("hidden channels (PCA → RGB)")

anim = FuncAnimation(fig, draw, frames=len(rgba_frames), interval=60)
anim.save(args.out, writer=PillowWriter(fps=16))
print(f"saved {args.out}")

# save PCA matrix for the web demo (3 x 12) + normalization stats
np.savez(args.weights,
         components=pca.components_,     # (3, 12)
         mean=pca.mean_,                 # (12,)
         scale=pca.explained_variance_ ** 0.5)  # (3,) rough per-component scale
print(f"saved {args.weights} for web export")
