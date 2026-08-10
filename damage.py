"""Damage/regeneration test: grow the heart, slice it in half, watch it heal.

    uv run python damage.py            # uses checkpoint.pt, writes damage.gif
"""
import numpy as np
import torch

from gnca import GraphNCA, alive_mask, seed_state

device = "mps" if torch.backends.mps.is_available() else "cpu"
ckpt = torch.load("checkpoint.pt", weights_only=False)
pos, edges = ckpt["pos"], ckpt["edges"].to(device)
N = pos.shape[0]

model = GraphNCA(channels=ckpt["channels"]).to(device)
model.load_state_dict(ckpt["model"])

GROW, HEAL = 80, 160
x = seed_state(1, N, model.channels, device,
               center=int(np.argmin(((pos - [0.5, 0.45]) ** 2).sum(1))))[0]

frames, cut_at = [], None
with torch.no_grad():
    for t in range(GROW + HEAL):
        frames.append(x[:, :4].cpu().numpy().copy())
        if t == GROW:
            cut_at = len(frames)
            # the slice: kill every cell in a horizontal band through the heart
            band = (pos[:, 1] > 0.42) & (pos[:, 1] < 0.58)
            x[ torch.from_numpy(band).to(device) ] = 0.0
        m = alive_mask(x, edges, N)
        x = model(x, edges) * m

import os
os.environ.pop("MPLBACKEND", None)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

fig, ax = plt.subplots(figsize=(5, 5), dpi=120)

def draw(i):
    rgba = frames[i]
    ax.clear()
    ax.scatter(pos[:, 0], pos[:, 1], c=np.clip(rgba[:, :3], 0, 1),
               s=14, alpha=np.clip(rgba[:, 3], 0, 1), linewidths=0)
    label = "growing" if i < cut_at else "healing"
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal",
           title=f"{label} (t={i})" + ("  *** SLICE ***" if i == cut_at else ""))
    ax.axis("off")
    if i >= cut_at:  # show where the cut was
        ax.axhspan(0.42, 0.58, color="red", alpha=0.06)

anim = FuncAnimation(fig, draw, frames=len(frames), interval=60)
anim.save("damage.gif", writer=PillowWriter(fps=16))
print(f"saved damage.gif (slice at frame {cut_at})")
