# GNNs as discretized PDEs: better local operators than isotropic diffusion

*Survey for the Graph NCA perception step. Written 2026-02-17. All links verified live.*

**Question.** Our rule is `x <- x + MLP([x, mean_n, mean_diff])` applied per node, where
`mean_diff_i = mean_{j in N(i)}(x_j - x_i)` is the random-walk graph Laplacian
(isotropic diffusion). What do the GNN-as-PDE papers use *instead* of plain isotropic
Laplacian smoothing, why, and what survives the NCA constraints (locality, weight
sharing, 50-100 rolled-out steps)?

**Short answer.** The entire literature replaces the *fixed, isotropic* Laplacian with a
*state-dependent, anisotropic* operator. The three workhorse ideas, in order of how
cheaply they map onto an NCA:

1. **Edge-stopping (Perona-Malik)**: multiply each edge's diffusive flow by a
   scalar `g(x_i, x_j)` that shrinks when the neighbor states differ — diffusion
   turns *off across sharp boundaries*. This is what GRAND's attention and BLEND's
   Beltrami flow implement, and it provably preserves gradient information that
   isotropic diffusion erases. One extra shared-MLP term per edge.
2. **Directional / advective terms**: signed, oriented flow (gradients, velocities,
   anti-symmetric couplings) that transports rather than averages. Advection
   (`x_j - x_i` with a *direction*) is the missing half of the classic CA
   Sobel-gradient perception.
3. **Reaction with multiple species**: activator/inhibitor pairs with *different*
   diffusion rates (Turing instability) turn the diffusion channel itself into a
   pattern generator rather than an eraser.

Everything below is cited; section 5 has measured numbers for *our* graphs.

---

## 1. The baseline: isotropic diffusion is an over-smoothing machine

GCN-style message passing is a discretization of the heat equation
`dx/dt = (P - I)x` where `P` is the random-walk transition matrix
(Chamberlain et al. 2021; see also the umbrella essay *Beyond Message Passing:
a Physics-Inspired Paradigm for Graph Neural Networks*, Bronstein et al., The
Gradient 2022, https://thegradient.pub/graph-neural-networks-beyond-message-passing-and-weisfeiler-lehman/).
Our `mean_diff` is exactly `(P - I)x`.

**Theory: decay is exponential in the number of steps.**

- **Oono & Suzuki**, *Graph Neural Networks Exponentially Lose Expressive Power for
  Node Classification*, ICLR 2020, https://arxiv.org/abs/1905.10947 — k steps of a
  GCN map inputs into a rank-deficient subspace at rate `O((s·lambda)^k)`, where
  `lambda` is the second-largest eigenvalue magnitude of the renormalized adjacency.
  Features collapse exponentially with depth.
- **Cai & Wang**, *A Note on Over-Smoothing for Graph Neural Networks*, 2020,
  https://arxiv.org/abs/2006.13318 — the Dirichlet energy `sum_edges ||x_i - x_j||^2`
  decays exponentially with layer count; the rate is controlled by the graph's
  spectral gap (how fast the graph mixes).
- **Keriven, Bietti, Vaiter**, *Not too little, not too much: a theoretical analysis
  of graph (over)smoothing*, NeurIPS 2022, https://arxiv.org/abs/2205.12156 —
  sharpens this: on random graphs with communities, states converge to the
  per-community average at rates given by specific spectral quantities; the
  collapse happens *within communities* at a finite, computable number of steps.

**Measured on our own graphs** (pure `mean_diff` iteration `x <- x + mean_diff`,
unit seed, no reaction — code at the bottom of this file):

| graph | steps until Dirichlet energy < 1% of start | variance after 100 steps |
|---|---|---|
| kNN random geometric, 1024 nodes, k=8 (project default) | ~10 | 1.4% |
| kNN random geometric, 1024 nodes, k=4 | ~10 | 1.6% |
| Watts-Strogatz ring, 1024, k=8, beta=0.05 | ~30 | 0.06% |
| regular 32x32 grid, 4-neighbor (classic CA topology) | ~180 | 1.3% |

On the kNN graphs the *first step alone* kills ~75-80% of the Dirichlet energy, and
the seed's height variance halves every 1-2 steps. On the classic pixel-grid
topology the collapse is slower (grid spectral gap ~ 2/L^2) but still ~99% gone
at 200 steps.

**Implication for an NCA that runs 50-100 steps:** after ~10-20 steps the
`mean_diff` *input channel* is at noise floor (0.1-1% of its initial magnitude) on
kNN topologies. The gradient information the rule needs to grow structure simply is
not in the isotropic channel anymore; any structure at step 50 is being carried by
the reaction term, which has to fight the fact that its only spatial-gradient
sensor is blind. This is the strongest argument in the literature for anisotropic
perception: anisotropic diffusion *keeps* the gradient channel alive for the whole
rollout.

---

## 2. What the papers use instead of isotropic Laplacian smoothing

### 2.1 Attention as diffusivity: the state-dependent Laplacian

**GRAND** — Chamberlain, Rowbottom, Gorinova, Webb, Rossi, Bronstein, *GRAND:
Graph Neural Diffusion*, ICML 2021, https://arxiv.org/abs/2106.10934.

- Model: `dx_i/dt = sum_{j in N(i)} a(x_i, x_j)(x_j - x_i)` — the heat equation with
  a *learnable, feature-dependent* diffusivity `a`. The paper calls it the
  "diffusivity" and builds it from attention:
  - scaled dot-product: `a(x_i,x_j) = softmax_j((W_K x_i)^T W_Q x_j / sqrt(d_k))`
  - or GAT-style: `a(x_i,x_j) = exp(LeakyReLU(v^T [W x_i || W x_j])) / sum_k ...`
- Three variants: **GRAND-l** (attention weights frozen during integration — a
  linear but *learned* diffusion operator), **GRAND-nl** (weights recomputed each
  integration step — genuinely nonlinear diffusion), **GRAND-nl-rw** (thresholds
  attention to rewire the graph).
- Multi-head: average of `h` attention heads.
- Their Figure 4 is exactly the NCA story: on a grid of pixels, GRAND-nl's
  attention weights act as an **edge detector**, "weighting diffusion within a class
  boundary in a way that preserves the image after diffusion," while plain
  Laplacian diffusion blurs it away.
- Key property for us: `a` is a *shared* function of a *pair* of node states —
  same weights at every node and every step. Locality and parameter sharing intact.

**GAT** — Veličković et al., *Graph Attention Networks*, ICLR 2018,
https://arxiv.org/abs/1710.10903 — is the same object viewed from the other side:
a single GAT layer computes exactly one forward-Euler step of nonlinear diffusion
with learned anisotropic coefficients. GATv2 (Brody, Alon, Yahav, ICLR 2022,
https://arxiv.org/abs/2105.14491) fixes the static-attention limitation of GAT
(dynamic attention — the weights can depend on the *joint* state of both nodes in
any configuration).

### 2.2 Beltrami flow: Perona-Malik diffusion with position

**BLEND** — Chamberlain, Rowbottom, Eynard, Di Giovanni, Dong, Bronstein,
*Beltrami Flow and Neural Diffusion on Graphs*, NeurIPS 2021,
https://arxiv.org/abs/2110.09443.

- Diffusion on the *joint* manifold of (node features, positional encodings).
- The scalar diffusivity is the classic Perona-Malik edge indicator
  `a = 1 / sqrt(1 + alpha^2 ||grad x||^2)`: **diffusion is weak exactly where the
  state gradient is large** — i.e., it sharpens boundaries instead of erasing them
  ("adaptive diffusion," citing Weickert 1998).
- Positional encodings are evolved *simultaneously* with features, so the graph
  topology itself flows (anisotropy through geometry, not only through features).
- On the NCA question: evolving positions is the *least* NCA-compatible part
  (topology changes break the fixed-edge assumption), but the edge-stopping
  coefficient is trivially local.

### 2.3 PDE taxonomy: parabolic vs hyperbolic vs elliptic layers

**PDE-GCN** — Eliasof, Haber, Treister, *PDE-GCN: Novel Architectures for Graph
Neural Networks Motivated by Partial Differential Equations*, NeurIPS 2021,
https://arxiv.org/abs/2108.01938.

- Treats layer depth as time and offers *three families* of dynamics:
  - **parabolic** (diffusion, `x' = L x`) — smoothing, the status quo;
  - **hyperbolic** (wave equation, `x'' = L x`) — *energy-conserving* oscillations,
    no dissipation, so no over-smoothing by construction;
  - **elliptic** (steady state) — control the endpoint, not the path.
- Learns the "speed" coefficients of each operator (including *per-channel* mixing
  matrices) and mixes the three dynamics in inception-style blocks; a learnable
  per-node gating chooses how much of each PDE each node gets.
- ADR-GNN (Eliasof, Haber, Treister, *Feature Transportation Improves Graph Neural
  Networks*, AAAI 2024, https://arxiv.org/abs/2307.16092) extends this to
  **advection-diffusion-reaction**: advection is a *transport* term — it moves
  feature mass along directed edges instead of averaging it, and is shown to
  improve over pure diffusion+reaction, especially on heterophilic data.

### 2.4 Energy-conserving / oscillatory dynamics

**A-DGN** — Gravina, Bacciu, Gallicchio, *Anti-Symmetric DGN: a stable architecture
for Deep Graph Networks*, ICLR 2023, https://arxiv.org/abs/2210.09789 — uses an
**anti-symmetric** coupling `(W - W^T)` (plus a small negative-definite damping)
so the ODE is stable but *non-dissipative*: the norm of the state is (near-)
conserved, so information survives dozens of layers.

**GraphCON** — Rusch, Chamberlain, Rowbottom, Mishra, Bronstein, *Graph-Coupled
Oscillator Networks*, ICML 2022, https://arxiv.org/abs/2202.02296 — second-order
dynamics `x'' = sigma(F(x)) - gamma x - alpha x'` (nonlinear coupled damped
oscillators). They *prove* that zero-Dirichlet-energy steady states (the
over-smoothed fixed point) are **unstable** for this system — the dynamics are
forced to keep oscillating. Any local coupling function `F` works (GCN or
attention), so this is a drop-in "engine" swap around an existing perception.

**GDE** — Poli, Massaroli, Park, Yamashita, Asama, *Graph Neural Ordinary
Differential Equations*, 2019, https://arxiv.org/abs/1911.07532 — the original
continuous-depth framing; NCA's 50-100 steps is literally a coarse ODE solver
rollout, and the GDE literature (e.g., 200-step solvers, per the GCDE experiments)
shows deep rollouts work *only* when the dynamics are stable/non-dissipative.

### 2.5 Reaction: the missing half

**CGNN** — Xhonneux, Qu, Tang, *Continuous Graph Neural Networks*, ICML 2020,
https://arxiv.org/abs/1912.00967 — `x' = (A - I)x + W x + b`: linear diffusion
*plus a learnable linear reaction/decay* (a "source" term that anchors states and
prevents total collapse).

**GRAND++** — Thorpe, Nguyen, Xia, Strohmer, Bertozzi, Osher, *GRAND++: Graph
Neural Diffusion with A Source Term*, ICLR 2022,
https://openreview.net/forum?id=EMxu-dzvJk — adds a fidelity term to GRAND
(attraction back toward the initial features). Provably keeps representations
non-collapsed even at low labeling rates.

**GREAD** — Choi, Hong, Park, Cho, *GREAD: Graph Neural Reaction-Diffusion
Networks*, ICML 2023, https://arxiv.org/abs/2211.14208 — a menu of reaction
equations (Fisher, Allen-Cahn, Zeldovich, blurring-sharpening, source, filter
banks) coupled to diffusion with a *learnable per-node diffusion coefficient*.
Reaction sharpens what diffusion blurs; the paper explicitly frames
reaction-diffusion as the balance between the two.

**RDGNN** — Eliasof, Haber, Treister, *Graph Neural Reaction Diffusion Models*,
SIAM J. Math. Data Sci. 2024, https://arxiv.org/abs/2406.10871 — the most NCA-
relevant paper: builds GNN layers from **Turing instability** (Section 4 below),
with *learned* diffusion coefficient matrices `Sigma` and a nonlinear reaction
`f`; uses implicit-explicit (IMEX) time integration; shows the reaction term
counteracts diffusion's eigenvalue-driven decay, mitigating over-smoothing, and
that the models work on heterophilic data precisely because the RD dynamics can
*generate* non-smooth patterns. Layers literally read
`U^{l+1} = U^l - kappa L~ U^l + k_U U^l`.

### 2.6 Opinion dynamics: bounded-confidence diffusion

**GODNF** — Hevapathige, Wijesinghe, Zehmakan, *Graph Neural Diffusion via
Generalized Opinion Dynamics*, 2025, https://arxiv.org/abs/2508.11249 — unifies
opinion-dynamics models (Deffuant-Weisbuch bounded confidence, Hegselmann-Krause)
into a trainable diffusion mechanism. The key idea: **diffusivity goes to zero
when opinions differ too much** — a principled, nonlinear, anisotropic
edge-stopping rule that provably avoids consensus (i.e., over-smoothing) and is
interpretable at depth.

### 2.7 Learnable weighted Laplacians and high-order diffusion

**HiD-Net** — Li, Wang, Liu, Shi, *A Generalized Neural Diffusion Framework on
Graphs*, 2023, https://arxiv.org/abs/2312.08616 — a unifying diffusion-with-
fidelity framework; identifies that most neural diffusion is *first-order* and
proposes high-order diffusion equations that are robust on heterophilic graphs.

**mu-ChebNet** — Zosso, Hariri, Kawasaki-Borruat, Berlureau, Vandergheynst,
*Geometry-Induced Diffusion on Graphs: A Learnable Weighted Laplacian for Spectral
GNNs*, 2026, https://arxiv.org/abs/2602.18141 — learns a *node-wise* weight
function `mu` that reshapes the Laplacian (edge conductivities) without touching
topology; spectral analysis shows the learned geometry picks preferred propagation
routes. Lightweight alternative to attention, and the learned weights are
interpretable.

---

## 3. Anisotropic vs isotropic diffusion, and where it comes from

- **Perona & Malik**, *Scale-space and edge detection using anisotropic
  diffusion*, IEEE TPAMI 12(7), 1990 — the original nonlinear diffusion:
  `dx/dt = div(g(|grad x|) grad x)` with edge-stopping functions such as
  `g(s) = exp(-(s/K)^2)` or `g(s) = 1/(1 + (s/K)^2)`. Smooths inside regions,
  *stops at edges*. This is the canonical answer to "what replaces isotropic
  diffusion": a scalar multiplier on each edge's flux that depends on the local
  gradient.
- **Weickert**, *Anisotropic Diffusion in Image Processing*, Teubner, 1998 — the
  standard reference; generalizes Perona-Malik to tensor-valued diffusivities
  (direction-dependent diffusion). GRAND and BLEND both cite it as their
  motivation.
- **GAT / attention-as-diffusivity**: on a graph there is no continuum
  gradient; the edge-stopping function becomes a *pairwise* function of
  `(x_i, x_j)`, which is exactly what attention computes. So "attention" and
  "learned edge conductivity" are the same object, and both are local and
  weight-shared (sections 2.1, 2.6).

---

## 4. Oscillatory reaction-diffusion and Turing patterns on graphs

- **Turing**, *The Chemical Basis of Morphogenesis*, Phil. Trans. R. Soc. B 237,
  1952 — activator-inhibitor systems with *different diffusion coefficients*
  spontaneously form stationary patterns (Turing instability): diffusion, the
  great homogenizer, becomes the pattern *generator* when species diffuse at
  different rates and react nonlinearly.
- **Nakao & Mikhailov**, *Turing patterns in network-organized activator-inhibitor
  systems*, Nature Physics 6:544, 2010, https://arxiv.org/abs/1005.1986 — the
  systematic treatment *on networks*: Turing conditions are expressed through the
  network Laplacian spectrum, and degree heterogeneity makes patterns much easier
  to excite than on regular lattices. Directly relevant to an NCA on a kNN graph:
  the graph's irregularity helps, not hurts, pattern formation.
- **Asllani et al.**, *The theory of pattern formation on directed networks*,
  Nature Communications 5:4517, 2014, https://arxiv.org/abs/1402.0760 — extends to
  directed edges (relevant if we ever want a directed, advective NCA).
- **RDGNN** (section 2.5) is the ML version: learn the diffusion coefficients and
  the reaction, and the layer can exhibit Turing instabilities — i.e., the *same*
  rolled-out dynamics that erase structure can be tuned to create it.

---

## 5. Over-smoothing: how many steps until collapse

See the measured table in section 1. Literature rates, translated into "steps":

- Oono & Suzuki (2020): rate `(s·lambda)^k` — exponential in layers; on graphs with
  a decent spectral gap, tens of layers collapse the representation.
- Cai & Wang (2020): Dirichlet energy decays exponentially at the graph's mixing
  rate. For a path/ring the mixing time is `O(L^2)` (slow), for expander-like
  graphs `O(log n)` (fast). Our kNN graphs sit near the fast end: ~10 steps to
  kill 99% of the gradient energy.
- Keriven et al. (2022): finite-step analysis — collapse happens *within
  communities* at rates computable from the spectrum; more steps than the
  spectral gap's inverse and the features are effectively constant per community.

**What this means for an NCA rolling out 50-100 steps:** the isotropic channel is
dead after ~10-20 steps on our topologies (kNN k=8), and mostly dead even on a
pixel grid (1-2% gradient energy left at 100 steps). The NCA survives only because
the reaction term re-creates the pattern; but every bit of spatial *perception*
the rule has must come from an operator that does not erase gradients. That is the
textbook argument for Perona-Malik / attention / advection perception: the
operator's job is to *report* differences, not to *average them away*.

---

## 6. Recommendations for the Graph NCA perception step

Constraint check first: all of the following are local (per-edge or per-node over
the 1-hop neighborhood) and weight-shared (same parameters at every node and every
step) — they fit the NCA contract.

1. **Edge-stopping gate on mean_diff (Perona-Malik, cheapest win).** Replace the
   fixed `mean_diff` with `mean_j [ g(x_i, x_j) * (x_j - x_i) ]` where
   `g = sigma(MLP_shared([x_i, x_j, |x_j - x_i|]))` (or the classic
   `1/(1+alpha^2 |x_j-x_i|^2)` with learned alpha). This is GRAND's attention,
   BLEND's edge indicator, and GODNF's bounded confidence, all in one local term.
   It keeps the diffusion channel alive for the full 50-100 step rollout.
2. **Signed/directional perception (advection).** Add the *directed* flux
   `mean_j [ (x_j - x_i) ]` *per sign* or per orientation — the graph analog of the
   Sobel gradient perception in the original Distill NCA (Mordvintsev et al.,
   *Growing Neural Cellular Automata*, Distill 2020,
   https://distill.pub/2020/growing-ca/), which the graph version dropped when it
   replaced pixel grids with kNN edges. ADR-GNN's advection term is the principled
   version: transport, not averaging. On a kNN graph, "up" and "down" edges are
   distinguishable by their positions — splitting mean_diff into
   `x_j - x_i` vs `x_i - x_j` accumulators, or weighting by neighbor *direction*,
   recovers orientational information with shared weights.
3. **Learnable per-channel conductivities (cheap, no pairwise function).**
   `mean_diff_w = mean_j [ diag(w_c(x_i)) * (x_j - x_i) ]` — a per-channel gate
   produced by the shared MLP (this is RDGNN's learned `Sigma`, PDE-GCN's learned
   speeds, mu-ChebNet's node-wise `mu`). Per-node (not pairwise) conductivities are
   isotropic but *state-dependent*, and already allow the rule to turn diffusion
   off selectively per channel.
4. **Activator-inhibitor split (Turing).** Hold two state groups with different
   (learned) diffusion conductivities and let the MLP react on the pair — the
   minimal Turing setup. RDGNN shows this turns the diffusion channel into a
   pattern generator and handles heterophilic structure.
5. **Second-order / oscillatory engine (if the MLP budget allows).** GraphCON's
   `x'' = MLP(perception) - gamma x - alpha x'` (implemented as a velocity state
   variable `v <- v + h(...)`, `x <- x + v`) provably avoids the over-smoothed
   fixed point; A-DGN's anti-symmetric coupling conserves norm. Momentum-like
   dynamics make 50-100 steps sustainable.
6. **Fidelity/source term.** A small pull toward the initial seed state
   (`+ kappa (x0 - x)`) — CGNN/GRAND++'s fix — guarantees the seed's signal
   survives the whole rollout; useful for the growth-from-seed task.
7. **Whatever you pick: verify the energy budget.** The right diagnostic is the
   Dirichlet energy of the *perception inputs* across a 100-step rollout: if the
   `mean_diff` input's energy collapses below ~1% before step 20, the perception is
   blind regardless of what the MLP does. The experiment script is below.

---

## Appendix: measured decay (reproducible)

```python
# pure mean_diff dynamics x <- x + (P-I)x from a unit seed; Dirichlet energy per step
import numpy as np
def diffusion_stats(pos, edge_index, steps=201, seed_node=0):
    e = edge_index.numpy() if hasattr(edge_index, 'numpy') else edge_index
    n = pos.shape[0]; src, dst = e[0], e[1]
    deg = np.bincount(dst, minlength=n).astype(float)
    x = np.zeros(n); x[seed_node] = 1.0
    D, V = [], []
    for _ in range(steps):
        mean_diff = np.zeros(n)
        np.add.at(mean_diff, dst, x[src] - x[dst]); mean_diff /= np.clip(deg, 1, None)
        x = x + mean_diff
        D.append(((x[src] - x[dst]) ** 2).sum()); V.append(x.var())
    return np.array(D), np.array(V)
```

---

## References (title — authors — year — link)

1. GRAND: Graph Neural Diffusion — Chamberlain, Rowbottom, Gorinova, Webb, Rossi, Bronstein — 2021 — https://arxiv.org/abs/2106.10934
2. Beltrami Flow and Neural Diffusion on Graphs (BLEND) — Chamberlain, Rowbottom, Eynard, Di Giovanni, Dong, Bronstein — 2021 — https://arxiv.org/abs/2110.09443
3. PDE-GCN: Novel Architectures for Graph Neural Networks Motivated by Partial Differential Equations — Eliasof, Haber, Treister — 2021 — https://arxiv.org/abs/2108.01938
4. Feature Transportation Improves Graph Neural Networks (ADR-GNN) — Eliasof, Haber, Treister — 2023/AAAI 2024 — https://arxiv.org/abs/2307.16092
5. Graph Neural Reaction Diffusion Models (RDGNN) — Eliasof, Haber, Treister — 2024 — https://arxiv.org/abs/2406.10871
6. GREAD: Graph Neural Reaction-Diffusion Networks — Choi, Hong, Park, Cho — 2023 — https://arxiv.org/abs/2211.14208
7. Graph-Coupled Oscillator Networks (GraphCON) — Rusch, Chamberlain, Rowbottom, Mishra, Bronstein — 2022 — https://arxiv.org/abs/2202.02296
8. Anti-Symmetric DGN — Gravina, Bacciu, Gallicchio — 2022/ICLR 2023 — https://arxiv.org/abs/2210.09789
9. Graph Neural Ordinary Differential Equations — Poli, Massaroli, Park, Yamashita, Asama — 2019 — https://arxiv.org/abs/1911.07532
10. Continuous Graph Neural Networks — Xhonneux, Qu, Tang — 2020 — https://arxiv.org/abs/1912.00967
11. GRAND++: Graph Neural Diffusion with A Source Term — Thorpe, Nguyen, Xia, Strohmer, Bertozzi, Osher — ICLR 2022 — https://openreview.net/forum?id=EMxu-dzvJk
12. Graph Attention Networks — Veličković, Cucurull, Casanova, Romero, Liò, Bengio — 2018 — https://arxiv.org/abs/1710.10903
13. How Attentive are Graph Attention Networks? (GATv2) — Brody, Alon, Yahav — 2022 — https://arxiv.org/abs/2105.14491
14. Diffusion Improves Graph Learning (GDC) — Gasteiger, Weißenberger, Günnemann — 2019 — https://arxiv.org/abs/1911.05485
15. Graph Neural Networks Exponentially Lose Expressive Power for Node Classification — Oono, Suzuki — 2020 — https://arxiv.org/abs/1905.10947
16. A Note on Over-Smoothing for Graph Neural Networks — Cai, Wang — 2020 — https://arxiv.org/abs/2006.13318
17. Not too little, not too much: a theoretical analysis of graph (over)smoothing — Keriven, Bietti, Vaiter — 2022 — https://arxiv.org/abs/2205.12156
18. A Generalized Neural Diffusion Framework on Graphs (HiD-Net) — Li, Wang, Liu, Shi — 2023 — https://arxiv.org/abs/2312.08616
19. Graph Neural Diffusion via Generalized Opinion Dynamics (GODNF) — Hevapathige, Wijesinghe, Zehmakan — 2025 — https://arxiv.org/abs/2508.11249
20. Geometry-Induced Diffusion on Graphs (mu-ChebNet) — Zosso, Hariri, Kawasaki-Borruat, Berlureau, Vandergheynst — 2026 — https://arxiv.org/abs/2602.18141
21. Scale-space and edge detection using anisotropic diffusion — Perona, Malik — IEEE TPAMI 1990
22. Anisotropic Diffusion in Image Processing — Weickert — Teubner 1998
23. The Chemical Basis of Morphogenesis — Turing — Phil. Trans. R. Soc. B 1952
24. Turing patterns in network-organized activator-inhibitor systems — Nakao, Mikhailov — Nature Physics 2010 — https://arxiv.org/abs/1005.1986
25. The theory of pattern formation on directed networks — Asllani, Busiello, Carletti, Fanelli, Planchon — Nature Communications 2014 — https://arxiv.org/abs/1402.0760
26. Beyond Message Passing: a Physics-Inspired Paradigm for Graph Neural Networks — Bronstein et al. — The Gradient 2022 — https://thegradient.pub/graph-neural-networks-beyond-message-passing-and-weisfeiler-lehman/
27. Growing Neural Cellular Automata — Mordvintsev, Randazzo, Niklasson, Levin — Distill 2020 — https://distill.pub/2020/growing-ca/
