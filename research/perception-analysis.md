# Perception-layer analysis (my notes)

Current perception: `z = [x, mean_n, mean_diff]` where `mean_diff = mean_n - x`.
- Redundant: third block is a linear combination of the first two; effective input is 2*channels.
- Fully isotropic: invariant to how neighbors are arranged around the node. Two neighborhoods
  with the same multiset of states but different geometry give identical perception.
  On a grid, Sobel-x/y exist precisely to break this symmetry.

Key fact about OUR setting: every graph comes with node positions
(k-NN on point clouds, ring lattices, mesh surfaces). We currently throw the geometry away.

## Options, ranked by fit

1. **Directional perception from relative positions (MeshNCA-style).**
   Pajouheshgar et al. 2024 (arXiv 2311.02820): weight each message by spherical harmonics of
   the neighbor's relative direction+distance — a continuous generalization of convolution
   kernels (Sobel is a lookup table over angle/distance; SH is the smooth version).
   Frozen, local, cheap, generalizes across graphs. Restores anisotropy.
   Cheaper variant: aggregate low-order directional moments,
   e.g. mean of unit(pos_j - pos_i) ⊗ (x_j - x_i) — a vector-valued gradient per channel.
2. **Distance-aware weights**: RBF of edge distance as per-edge conductivities
   (anisotropic diffusion, Perona-Malik flavor). Frozen or learned.
3. **Multi-hop powers** (A, A^2 means, ChebNet-style): widens receptive field per step.
   Cheap but the NCA iterates anyway, so information already spreads; least urgent.
4. **Sum vs mean** (GIN vs GraphSAGE): mean loses degree/cardinality info; sum is maximally
   expressive (WL) but degree-dependent, which hurts generalization across graphs/densities.
   Could add degree as a scalar feature instead — one number, same benefit, no instability.
5. **Attention (GAT-lite)**: learned per-edge weights from states. Most params, least NCA-spirit,
   plausible instability for a rule iterated 64-120 steps. Try only after 1-2.

## Caveat on expressiveness theory
WL/expressivity results are about one-shot readout GNNs. An NCA is a dynamical system:
the rule iterates, information diffuses over time, and the fixed point is what matters.
The binding constraints are anisotropy and training stability, not WL power.
