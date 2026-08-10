# Prior work: Neural Cellular Automata on graphs and non-Euclidean domains

Research survey for our project: a Graph Neural Cellular Automaton (GNCA) — a shared tiny MLP update rule on nodes, with perception `[own state, mean of neighbor states, mean of (neighbor − own)]`, stochastic residual updates, trained Growing-NCA-style (Mordvintsev et al., 2020) to grow a target node-coloring pattern from a single seed.

Date: 2026-07. Compiled from web searches; every paper's abstract/PDF/HTML was fetched and read (details below come from the papers themselves, not from memory).

---

## 1. The reference setup (grid NCA) — what "growing a pattern" means

**Growing Neural Cellular Automata** — Mordvintsev, Randazzo, Niklasson, Levin. *Distill*, 2020. https://distill.pub/2020/growing-ca

- State per cell: `[R,G,B,A]` visible + hidden channels (16 total). One seed cell, everything else zero.
- **Perception (fixed, hand-designed):** per-channel 3×3 **Sobel X, Sobel Y, Laplacian** convolutions, concatenated with own state → perception vector `[s, Kx*s, Ky*s, Klap*s]`.
- **Update (learned):** small conv net (3×3 → 1×1), residual: `s ← s + tanh(net(p)) ⊙ mask`, where `mask` is Bernoulli(0.5) — **stochastic/asynchronous updates**, plus an **alive mask** (alpha channel > 0.1).
- **Training:** sample pool of 1024 states; each step draw a batch, evolve each sample a random number of steps (~64–96, or fixed), compute L1+L2 against target on visible channels, occasionally inject the seed into the pool and **damage** pool samples (regeneration robustness).
- The Sobel filters are what give a cell **direction information** ("which way is the gradient"). This is the single thing that does not transfer to graphs: there is no canonical "up/right" on an arbitrary graph.

Direct antecedents for "learn a CA rule with a neural net": Wulff & Hertz, *Learning Cellular Automaton Dynamics with Neural Networks*, NeurIPS 1992 (CNNs with shared weights); Gilpin, *Cellular automata as convolutional neural networks*, Phys. Rev. E 2019 (https://arxiv.org/abs/1809.02942) — any local CA rule can be realized by a CNN with fixed small kernels.

**Self-Organising Textures** — Niklasson, Mordvintsev, Randazzo, Levin. *Distill*, 2021. https://distill.pub/selforg/2021/textures — trains NCA to synthesize stochastic textures using a style/perceptual loss (VGG feature statistics); confirms pooling + stochastic updates are the key training machinery; note "these can be defined for most … manifolds".

---

## 2. Graph NCA — the core literature

### 2.1 Learning Graph Cellular Automata (the GNCA paper)
**Grattarola, Livi, Alippi.** NeurIPS 2021. https://arxiv.org/abs/2110.14237 | PDF: https://proceedings.neurips.cc/paper/2021/file/af87f7cdcda223c41c3f3ef05a3aaeea-Paper.pdf | Code: https://github.com/danielegrattarola/GNCA | Author summary: https://danielegrattarola.github.io/posts/2021-11-08/graph-neural-cellular-automata.html

First systematic study of learning GCA transition rules with GNNs. Formalizes a GCA as `(G, S, N, τ)`; proves a GNCA with one-hot preprocessing + pattern matching can represent **any** finite/discrete GCA.

- **Perception/message passing:** a message-passing block inspired by the GNN design space (You et al. 2020): `h_i' = h_i ‖ Σ_{j∈N(i)} ReLU(W h_j + b)`, i.e. **own state concatenated with the sum of ReLU-transformed neighbor states**, surrounded by pre- and post-processing MLPs. Aggregation is a plain (isotropic) **sum** — no directions.
- **Learned vs fixed filters:** everything in the message-passing block is learned. The perception is *not* a fixed-kernel "Sobel on graphs".
- **Anisotropy handling (the paper's own answer):** formalizes anisotropy as dependence on **edge attributes** `e_ij` (direction, distance, unique ID of the relation). Two concrete techniques: (1) **edge-conditioned convolution** — weight matrix `W` computed by a kernel-generating network from `e_ij`; (2) simply **concatenating `e_ij` to `h_j`** in the message.
- **Experiments:** (a) learn a Voronoi/Delaunay outer-totalistic binary GCA (supervised 1-step, reaches 100% accuracy, autonomous rollouts don't diverge); (b) imitate Boids flocking (continuous states, *dynamic* graph); (c) **morphogenesis: converge to a target point cloud** (grid, Stanford bunny, Minnesota road network, PyGSP logo, Swiss roll) — this is the experiment closest to ours.
- **Training tricks for morphogenesis:** BPTT over `t` steps with loss `MSE(τ^t(S), Ŝ)`; random `t ∈ [10, 20]` per forward pass ("as also done by Mordvintsev et al."); a **replay cache of 1024 states** — after each forward pass the reached state is stored, batches are sampled from the cache, and **one cache entry is replaced with the seed** `S̄` to avoid catastrophic forgetting and to teach persistence at the target. Fixed `t=10` or `t=20` tends to learn *periodic orbits* around the target rather than convergence; random `t∈[10,20]` reliably learns a stable attractor. **No stochastic updates, no alive mask** in the paper's GNCA.
- Important negative result to know: even with a good loss, the learned rule often oscillates around the target instead of settling — periodic attractors are the default failure mode; random rollout lengths fix it.

### 2.2 E(n)-equivariant Graph Neural Cellular Automata
**Gala, Grattarola, Quaeghebeur.** arXiv 2023 (2301.10497). https://arxiv.org/abs/2301.10497 | HTML: https://arxiv.org/html/2301.10497v2 | Code: https://github.com/gengala/egnca

Argues existing GNCAs (Grattarola et al.) can violate CA locality by using global information and are **anisotropic** (not equivariant to isometries of node coordinates). Replaces standard graph convolutions with EGNN-style E(n)-equivariant ones.

- **Perception/message passing:** messages `m_ij = φ_m(h_i, h_j, ‖x_i − x_j‖)` — the *relative distance* is the only geometric information; aggregation is a plain **sum**; then `h_i' = φ_h(h_i, Σ_j m_ij)`; coordinates update as `x_i' = x_i + Σ_j (x_i − x_j) φ_x(m_ij)` (EGNN, Satorras et al. 2021).
- **Fixed vs learned:** all MLPs learned (`φ_m: R^{2h+1} → R^m`, TanH, 5.3k params total; hidden dim 16, message dim 32). Perception is learned but *isotropic by construction* — only distances, never absolute positions, enter the computation.
- **Hidden states:** unlike Grattarola (state = node coordinates only), they keep per-node hidden feature vectors `H` (initialized to 1s), which encode history and orchestrate morphogenesis — "all nodes share the same genome."
- **Training tricks (very relevant):** (a) **PairNorm** (Zhao & Akoglu 2020) applied to node features after every step — parameter-free normalization that fights over-smoothing/vanishing gradients in the recurrent setting; (b) TanH activations everywhere; (c) residual/skip connection; (d) **state pool per graph** with batch size ramped 4→32; (e) Adam lr 5e-4, reduce-on-plateau, gradient clipping, weight decay; (f) **E(n)-invariant loss**: instead of MSE to target coordinates, minimize squared error between *all pairwise distances* `‖x_i' − x_j'‖` and target distances — needed because the model never sees global orientation. They train to convergence and demonstrate regeneration after damage (open-ended persistency, 1000+ steps).
- Lessons: normalize hidden state every step; use invariant losses if your target has a symmetry group; strict locality (single message-passing layer) is a deliberate feature, the receptive field grows via iteration.

### 2.3 GNCA applications (shows the design space is being reused)
- **Training Topology With Graph Neural Cellular Automata** — Dwyer et al., IEEE eIT 2023. https://ieeexplore.ieee.org/document/10187381 — uses GNCA-style dynamics to evolve/train sparse neural-network topologies.
- **Physics-Informed Graph Neural Cellular Automata: An Application to Compartmental Modelling** — Navarin et al., IJCNN 2024. https://ieeexplore.ieee.org/document/10650578 — GNCA as an interpretable, physics-informed (SINDy-style) epidemic model.
- **Spatiotemporal modeling with Graph Neural Cellular Automata for Modular Traffic Forecasting** — Astore et al., Expert Systems with Applications 2026. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6408268 — modular GNCA traffic forecasting.
- **Implementing Graph Neural Cellular Automata** — Louis & Parker, Stanford CS224W blog. https://medium.com/stanford-cs224w/implementing-graph-neural-cellular-automata-098db0bca011 — student implementation + tutorial of Grattarola's architecture.

### 2.4 Developmental Graph Cellular Automata (growing the graph itself)
**Waldegrave, Stepney, Trefzer.** ALIFE 2023. https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00658 | PDF: https://eprints.whiterose.ac.uk/id/eprint/203478/1/isal_a_00658.pdf

Grows *directed graphs* from a single node, with the update rule encoded in neural weights (evolved via random search in the paper, not gradient-trained).

- **Perception (fixed, isotropic by requirement):** with one-hot node states `S`, the preprocessing computes (1) **counts of neighbor states** `C = A·S` (in/out variants using `A` and `Aᵀ`), and (2) **graph-Laplacian filtering** `F = L·S` with directed "out-Laplacian" `D_out − A` and "in-Laplacian" `D_in − Aᵀ`. So their "perception" ≈ [counts of neighbors per class, Laplacian of neighbors] — note `L·S` is exactly a sum of `(state_j − state_i)` terms, i.e. a scaled version of our `mean(neighbor − own)`.
- **Anisotropy handling:** with an unordered, variable-degree neighborhood the update function *must* be permutation invariant; the only anisotropy they admit is **edge direction** — incoming vs outgoing neighbors are two distinct types (a graph-native substitute for "left vs right").
- **Update:** single-layer perceptron over `[F_out, F_in, 1]` outputs one of `S+2` actions: change state, duplicate node (copies all connections), delete node. Growth **naturally halts** in an attractor — a nice property for graph generation.
- Lesson: degree-free fixed-size perception = counts + Laplacian terms; direction-typed edges give cheap anisotropy.

### 2.5 NCA for growing networks / circuits (graph-adjacent)
- **HyperNCA: Growing Developmental Networks with Neural Cellular Automata** — Najarro, Sudhakaran, Glanois, Risi. ICLR 2022. https://arxiv.org/abs/2204.11674 — NCA as a hypernetwork that grows the weight matrix of a policy network on a grid of "weight cells" (grid-based, but the idea of NCA-as-developmental-generator for structured objects transfers).
- **Self-Organising Digital Circuits** — Barylli, Béna, Mordvintsev, Nisioti, Risi. arXiv 2026 (2608.02606). https://arxiv.org/abs/2608.02606 — NCA-pattern-generation paradigm lifted to graphs: a topology-masked Transformer configures LUTs of a Boolean circuit so that the system self-assembles functional circuits and re-routes logic around faults. Confirms the graph+NCA pattern-generation framing is active research.

---

## 3. Meshes, surfaces, particles (non-Euclidean, not graphs-per-se)

### 3.1 Mesh Neural Cellular Automata (closest to our perception)
**Pajouheshgar, Xu, Mordvintsev, Niklasson, Zhang, Süsstrunk.** ACM TOG 2024. https://arxiv.org/abs/2311.02820 | HTML: https://arxiv.org/html/2311.02820v2 | Project: https://meshnca.github.io/

Synthesizes dynamic textures directly on 3D mesh vertices, no UV maps; trained on one icosphere, generalizes to unseen meshes; runs in WebGL.

- **Perception ("Mesh Perception", fixed/non-parametric):** generalizes the grid's 3 fixed filters to meshes by writing a 2D convolution filter as a function of the neighbor's polar angle and distance, then evaluating the same idea with **spherical harmonics** `Y_l^m(θ_ij, φ_ij)` of the *relative direction* of neighbor `j` from cell `i` (first order, `l ≤ 1`, 4 basis functions; distance dependence dropped — found unnecessary). Message: `Φ(i←j) = w_ij (s_j − s_i)`, perception = sum over neighbors. `Y_0^0` is the **Laplacian** (their ablation: this alone is isotropic, like our mean(neighbor−own)); `Y_1^m` are the **Sobel analogs** (directional gradients in 3D). Explicitly **not learned** — the authors ablate learned vs fixed perception and keep fixed.
- **Adaptation (learned):** per-vertex MLP, 2 layers, ReLU, input `[s_i, z_i, h_i]` (h = optional per-cell conditioning, e.g. motion vector), residual update masked by **Bernoulli(0.5)** per vertex (asynchronicity, same as Growing NCA).
- **Training tricks:** differentiable renderer + **VGG style loss** (relaxed Wasserstein distance + moment matching, Kolkin et al. 2019) for image-guided synthesis; **pool** of 256 (image) / 64 (text) samples with **seed re-injection every 16/32 epochs**; random **step range [15, 25]** per epoch ([32, 64] with motion targets); **overflow loss** `Σ|S − clip(S, [-1,1])|` weighted 10000 to keep states bounded; multi-step LR decay. **Grafting** = spatial/temporal interpolation between two trained instances (train with "compatible" initialization so states are in the same coordinate system).
- Lessons: fixed isotropic + fixed directional filters can be decoupled; on meshes the natural "directions" are spherical harmonics of relative positions; overflow loss and pool injection frequencies are concrete, copyable numbers.

### 3.2 3D Texture Synthesis Using Graph Neural Cellular Automata
KTH master's thesis, 2023. https://www.diva-portal.org/smash/get/diva2:1787727/FULLTEXT01.pdf — GNCA + differentiable renderer for 3D mesh texture synthesis; the thesis contains a survey chapter of NCA-into-non-Euclidean-domains work (2D/3D grids, graphs) worth mining for more citations.

### 3.3 Neural Particle Automata (Lagrangian NCA)
**Kim, Pajouheshgar, Süsstrunk, Jakob, Park.** arXiv 2026 (2601.16096). https://arxiv.org/abs/2601.16096 | Project: https://selforg-npa.github.io/

NCA where cells are particles with continuous positions (Lagrangian, unlike Eulerian grids/meshes); neighborhoods are dynamic, built from **Smoothed Particle Hydrodynamics (SPH)** operators.

- **Perception:** `[S_i, S̃_i, ∇S_i, ∇ρ_i]` — own state, SPH kernel-weighted neighborhood average (value-like), SPH gradient (derivative-like, = directional info), and **density gradient** (geometric cue about local crowding — a graph analog: degree!).
- **Stability tricks (transferable):** (1) **log-compression of vector perception terms**: `v ← log(1+|v|) · v/(|v|+η)` — prevents huge gradient magnitudes in recurrent training; (2) **stop-gradient through positions** during SPH perception (helps stability, enables bigger LR); (3) **displacement regularization** `Σ_t ‖Δx(t)‖` to prevent jitter; (4) variable step sampling + state pool + overflow regularizer, "standard NCA practices."
- Tasks: morphogenesis (4096–16384 particles), particle textures, **point-cloud classification** (static point set, iterative local communication) — i.e. NCA-as-GNN on point clouds works.
- They note rotational equivariance is *not* enforced but could be via Vector Neurons (Deng et al. 2021) on gradient inputs.

---

## 4. Differentiable / attention-based CA variants (grid, but idea-rich)

- **Growing Isotropic Neural Cellular Automata (IsoNCA)** — Mordvintsev, Randazzo, Fouts. ALIFE 2022. https://arxiv.org/abs/2205.01681 — removes Sobel X/Y, keeps only the **Laplacian** (perception = own state + per-channel Laplacian), update `s += relu(pW0+b0)W1`, stochastic p=0.5, alive mask A>0.1. To still grow asymmetric patterns they either use **structured seeds** (3+ non-collinear, distinctly encoded points define orientation) or a **rotation-reflection-invariant loss** (match target under best rotation/reflection, computed efficiently via polar coordinates + FFT). Key quote: with isotropic perception, orientation must come from *elsewhere* — seed structure or symmetry-breaking noise (asynchronous updates).
- **Growing Steerable Neural Cellular Automata** — Randazzo, Mordvintsev, Niklasson, Levin. ALIFE 2023. https://arxiv.org/abs/2302.10197 — instead of dropping direction, each cell carries an **internal orientation state** and rotates its perceived gradient by it ("cells can turn"). This restores directionality while remaining globally rotation-invariant; chirality emerges. Transferable idea: an explicit per-cell orientation channel is a way to get anisotropic behavior on an isotropic substrate.
- **Attention-based Neural Cellular Automata (ViTCA)** — Tesfaldet, Nowrouzezahrai, Pal. NeurIPS 2022. https://arxiv.org/abs/2211.01233 — replaces the fixed convolution perception with **spatially localized self-attention**: attention masked to each cell's neighborhood (O(NMd) instead of O(N²d)); optional **positional encoding** concatenated to each cell state; learned attention + MLP; residual; pool-sampling training. Best denoising results vs UNetCA/ViT at matched parameter count. Lesson: *learned* perception (attention over the neighborhood) works and beats fixed filters on grids — a candidate for graphs (attention over neighbor set with edge features).
- **Neural Cellular Automata Manifold (NCAM)** — Ruiz, Vilalta, Moreno-Noguer. ICLR 2021. https://arxiv.org/abs/2006.12155 — wraps the NCA update in a hypernetwork: **dynamic convolutions** whose kernels are generated from a latent code, inside an autoencoder, learning a manifold of NCA rules ("transcription factors" analogy). Transferable: conditioning the update rule on a per-instance code.
- **Latent Neural Cellular Automata (LNCA)** — Menta et al. arXiv 2024 (2403.15525). https://arxiv.org/abs/2403.15525 — runs the NCA in the latent space of a pretrained autoencoder; up to 16× larger inputs at the same budget. Transferable: if our node states are high-dimensional or graphs are huge, run the CA in a compressed per-node space.
- **Variational Neural Cellular Automata** — Palm, González-Duque, Sudhakaran, Risi. ICLR 2022. https://openreview.net/forum?id=7fFO4cMBx_9 — generative-model treatment of NCA (latent code per sample, KL regularization).
- **Pathfinding Neural Cellular Automata** — Earle et al. 2023. https://openreview.net/forum?id=CU8BwVAzLme — hand-coded and learned NCA rules for pathfinding; algorithmic alignment analysis.
- **AdaNCA** — NeurIPS 2024. https://neurips.cc/virtual/2024/poster/96193 — NCA as adaptors between ViT layers (grid).
- **A New Kind of Network? Review and Reference Implementation of NCA (NCAtorch)** — Spitznagel & Keuper. arXiv 2026 (2604.24990). https://arxiv.org/abs/2604.24990 — a survey + reference framework; modular **perception modules** (conv, attention, Sobel, deformable, residual), sample-pool mechanism, ablation showing pool impact on regeneration. Good starting bibliography; grid-centric.

---

## 5. Perception operators on graphs — synthesis

| Work | Domain | Perception (what a cell sees) | Fixed or learned | Directional info? |
|---|---|---|---|---|
| Growing NCA 2020 | grid | `[s, SobelX*s, SobelY*s, Laplacian*s]` | fixed | yes (Sobel) |
| IsoNCA 2022 | grid | `[s, Laplacian*s]` | fixed | no (by design) |
| SteerableNCA 2023 | grid | `[s, SobelX,Y rotated by own orientation, Laplacian]` | fixed filters + learned orientation | yes, via internal angle |
| ViTCA 2022 | grid | masked local self-attention over neighborhood (+PE) | learned | yes (attention + PE) |
| Grattarola GNCA 2021 | graph | `s_i ‖ Σ ReLU(W s_j + b)` (pre/post MLPs) | learned | only via edge attributes |
| E(n)-GNCA 2023 | graph (w/ coords) | `φ_m(s_i, s_j, ‖x_i−x_j‖)`, sum-aggregate | learned | only via distances (isotropic) |
| MeshNCA 2024 | mesh | `Σ_j Y_l^m(θ_ij,φ_ij)·(s_j − s_i)`, l≤1 (4 SH filters) | fixed (non-parametric) | yes (SH = Sobel analog) |
| NPA 2026 | particles | `[S_i, S̃_i, ∇S_i, ∇ρ_i]` (SPH) | fixed operators, learned MLP | yes (gradients) |
| DGCA 2023 | growing directed graph | counts `A·S`, Laplacian `L·S` (in/out variants) | fixed | edge direction only |

Observations:
- **Isotropic aggregation (sum/mean) is the default** on graphs; nobody uses a fixed "directional filter" on an arbitrary graph because there is no global direction to orient it.
- The **only papers that get real directionality on non-grid domains** do it through *relative geometry*: coordinates (E(n)-GNCA distances; MeshNCA SH angles; NPA SPH gradients) or *edge types* (DGCA in/out). If our graphs carry node coordinates, spherical-harmonic-of-relative-position filters are the proven choice (MeshNCA). If not, we need positional encodings (below).
- **Our perception `[own, mean(neighbor), mean(neighbor−own)]` is a graph-native analog of `[s, Laplacian*s]`** (mean(neighbor−own) ≈ Laplacian scaled by degree, exactly what DGCA and MeshNCA use). The `mean(neighbor)` term is extra (a "neighborhood context" channel that Laplacian-only models lack); it also makes the perception degree-normalized, which Grattarola's sum-aggregation is not.

---

## 6. The anisotropy problem: every solution in the literature

The grid gets directions from Sobel kernels. On a graph there is no canonical direction, so prior work either gives up direction, imports it, or synthesizes it:

1. **Give up direction (isotropy by construction).** Sum/mean aggregation (Grattarola; E(n)-GNCA); Laplacian-only perception (IsoNCA, MeshNCA's Y₀⁰, DGCA). Cost: any asymmetric target needs another symmetry-breaking mechanism.
2. **Break symmetry from the seed.** Structured/multi-point seeds with distinct channel encodings (IsoNCA: 3 non-collinear points; SteerableNCA: 2 seeds). Directly applicable to us: seed a few nodes with distinct "anchor" channels to fix orientation of the grown pattern.
3. **Break symmetry via noise/asynchrony** + a symmetry-invariant loss (IsoNCA's rotation-reflection-invariant objective via polar/FFT; SteerableNCA). On graphs the analogous trick is a loss that matches the target up to graph automorphisms, or up to rotation if coordinates exist (E(n)-GNCA's all-pairs-distance loss is exactly this for E(n)).
4. **Import geometry as edge attributes.** Edge-conditioned convolutions or concatenated `e_ij` (Grattarola); relative distances (E(n)-GNCA); spherical-harmonic angles (MeshNCA); SPH gradients (NPA). Best when coordinates are meaningful.
5. **Edge direction as the only anisotropy** (DGCA in/out Laplacians) — works on directed graphs with zero geometric assumptions.
6. **Positional encodings (GNN literature, not NCA-specific but directly transferable):**
   - **Laplacian eigenvectors** as node features — the standard graph substitute for coordinates: Dwivedi et al., *Benchmarking Graph Neural Networks* (2023, arXiv:2003.00982); Kreuzer et al., *Rethinking Graph Transformers with Spectral Attention* (NeurIPS 2021, arXiv:2106.03893).
   - **Directional Graph Networks** — Beaini et al., ICML 2021, arXiv:2010.02863: use Laplacian eigenvectors to define *direction vector fields* per node, then aggregate messages separately per direction — literally "Sobel on graphs" via eigenvectors. This is the single most on-point anisotropy hack for our setting: augment the MLP input with per-neighbor directional projections `⟨e_i, x_j − x_i⟩`-style terms using eigenvector-derived axes.
   - **Position-aware GNNs** — You, Ying, Leskovec, 2019 (arXiv:1906.04817): distances to anchor sets as positional features.
   - **Principal Neighbourhood Aggregation (PNA)** — Corso et al., NeurIPS 2020 (arXiv:2004.05718): multiple aggregators (mean/max/min/std) scaled by degree — directly relevant because our mean-based perception is degree-normalized; PNA says *combine* aggregators and rescale by `log(degree)` to handle degree heterogeneity.
7. **Learn the perception.** Attention over the neighborhood with learned keys/queries (ViTCA on grid; on graphs this is just a graph attention layer (GAT, Veličković et al. 2018) used recurrently). E(n)-GNCA effectively learns messages `φ_m(s_i, s_j, d_ij)`.

---

## 7. Training tricks that matter (with numbers)

1. **State pool / replay cache** — the single most-repeated mechanism. Growing NCA: 1024 states, batch of 8, seed injection + damage. Grattarola: 1024-state cache, one entry replaced by seed each batch (anti-forgetting). E(n)-GNCA: per-graph pool, batch size ramped 4→32. MeshNCA: pool 256 (image) / 64 (text), seed every 16/32 epochs. NCAtorch: configurable pool ratio, perturbed samples.
2. **Random rollout length per forward pass** — Growing NCA ~64–96; Grattarola `t∈[10,20]` (fixes periodic-orbit failure); MeshNCA `[15,25]` / `[32,64]`. Random `t` is what makes the target a stable attractor instead of an orbit.
3. **Stochastic / asynchronous updates** — Bernoulli(0.5) masking (Growing NCA, IsoNCA, MeshNCA). Not used by Grattarola/E(n)-GNCA (they keep synchronous updates), so it's optional but strongly associated with robustness and (in IsoNCA) with symmetry breaking. We already do this.
4. **Alive masking** — grid-only (alpha > 0.1); requires an explicit "alive" channel and a notion of empty cells. Transferable to graphs if we add a per-node alpha channel and freeze non-alive nodes (useful for growing patterns where nodes should stay "off").
5. **Overflow loss** — `Σ|S − clip(S, [-1,1])|`, weight 10⁴ (MeshNCA); keeps recurrent states bounded. Directly applicable.
6. **Normalization for recurrence** — PairNorm (or NodeNorm) after every step + TanH activations + residual connections (E(n)-GNCA). Prevents over-smoothing/exploding states in long rollouts. Very relevant for us.
7. **Invariant losses** — E(n)-GNCA: all-pairs distance MSE instead of coordinate MSE. IsoNCA: min-over-rotations pixel loss via polar FFT. On graphs: match up to automorphism, or use pairwise distances if coordinates exist.
8. **Damage augmentation** — damage pool samples during training (Growing NCA) → regeneration at test time; E(n)-GNCA damages at t'=25 and runs 1000 extra steps.
9. **Gradient hygiene** — log-compress vector perception terms (NPA), stop-gradients through geometry, displacement regularization, gradient clipping + weight decay (E(n)-GNCA), LR reduce-on-plateau, batch-size ramps.
10. **Per-cell conditioning** — MeshNCA's `h_i` (motion fields); NCAM's dynamic kernels; VNCA's latent codes. Gives one rule, many behaviors.
11. **Grafting/interpolation** — MeshNCA: train instances to be compatible, then interpolate states/weights at test time; a nice demo property for node-coloring patterns too.

---

## 8. Concrete recommendations for our GNCA

Ranked by expected payoff for "shared tiny MLP, perception = [own, mean(neighbor), mean(neighbor−own)], stochastic residual updates, grow a target coloring from one seed":

1. **Keep mean-based, degree-normalized perception; add the Laplacian interpretation on purpose.** `mean(neighbor − own)` is the graph analog of the Laplacian filter that IsoNCA/MeshNCA/DGCA all converge on; `mean(neighbor)` adds context. Both are permutation-invariant and degree-robust — the right isotropic default. (Grattarola's sum aggregation is *not* degree-normalized and will behave differently on high-degree nodes.)
2. **Add an explicit per-node "alive"/growth channel and an alive mask** (Growing NCA / IsoNCA style: A > 0.1). This is the standard mechanism for "grow from a seed and stop": empty nodes freeze, which also stabilizes training. Needs a dead-node convention for graph neighborhoods.
3. **Fix periodic-orbit failure by sampling rollout length per batch** (Grattarola: `t ∈ [10, 20]`; scale to graph diameter; e.g. `t ∈ [2d, 4d]` where d = diameter). This is the empirically proven cure for "converges then oscillates."
4. **Use a state pool with seed re-injection (and damage).** Copy Grattarola's exact scheme: cache of reached states, replace one entry with the seed each step; or MeshNCA's periodic re-injection. This buys persistence and regeneration.
5. **For asymmetric targets on unpositioned graphs: structured seed + symmetry-breaking asynchrony.** Seed several nodes with distinct "anchor" channels (IsoNCA's 3-point seed idea, adapted: pick 3+ nodes whose relative positions in the graph define orientation). If the target is invariant up to automorphism, use a min-over-automorphism loss; with coordinates, use E(n)-GNCA's pairwise-distance loss.
6. **If coordinates exist (or can be computed: spectral layout, spring layout), add directional perception — the MeshNCA recipe:** per-neighbor weight `w_ij` from a fixed basis over the relative direction vector (spherical harmonics on 3D; for planar layouts just cos/sin of the angle — a graph Sobel), message `w_ij (s_j − s_i)`, concatenated into the perception. Keep it fixed/non-parametric (MeshNCA's ablation says fixed ≥ learned here, and it costs nothing at inference).
7. **If coordinates don't exist, use Laplacian-eigenvector positional encodings as node channels** (Dwivedi-style LapPE; Beaini's Directional Graph Networks for actual directional aggregation: project `x_j − x_i`-like edge vectors onto eigenvector axes and aggregate per axis — this is the closest thing to Sobel on graphs).
8. **Consider edge-direction as a cheap anisotropy axis** if the graph is directed (DGCA in/out Laplacians).
9. **Stabilize the recurrence:** normalize node states every step (PairNorm/NodeNorm), TanH or bounded activations, overflow loss at weight ~10⁴, gradient clipping. These are the E(n)-GNCA + MeshNCA staples and directly attack the "states explode after 100 steps" failure.
10. **Log-compress large-magnitude perception terms** (NPA): `v ← log(1+|v|) v/(|v|+η)` — cheap insurance for degree-heterogeneous graphs.
11. **Think of the MLP as the only adaptive part; keep perception fixed.** Every graph/mesh/particle work either fixes perception (MeshNCA, DGCA) or learns it as part of the message MLP (E(n)-GNCA, Grattarola). Our current fixed [own, mean, mean-diff] + tiny MLP sits exactly in this design space; the upgrade path is *add features to the perception vector*, not a bigger MLP.
12. **For richer targets, one step further:** localized graph attention as perception (ViTCA→GAT), latent-space NCA for huge graphs (LNCA), per-instance conditioning codes (NCAM/VNCA), and grafting for interpolation between patterns (MeshNCA).

---

## 9. Reference list (all links verified during this survey)

**Graph NCA**
- Grattarola, Livi, Alippi. *Learning Graph Cellular Automata*. NeurIPS 2021. https://arxiv.org/abs/2110.14237 · code https://github.com/danielegrattarola/GNCA
- Gala, Grattarola, Quaeghebeur. *E(n)-equivariant Graph Neural Cellular Automata*. 2023. https://arxiv.org/abs/2301.10497 · code https://github.com/gengala/egnca
- Waldegrave, Stepney, Trefzer. *Developmental Graph Cellular Automata*. ALIFE 2023. https://direct.mit.edu/isal/article/doi/10.1162/isal_a_00658
- Dwyer et al. *Training Topology With Graph Neural Cellular Automata*. IEEE eIT 2023. https://ieeexplore.ieee.org/document/10187381
- Navarin et al. *Physics-Informed Graph Neural Cellular Automata: An Application to Compartmental Modelling*. IJCNN 2024. https://ieeexplore.ieee.org/document/10650578
- Astore et al. *Spatiotemporal modeling with Graph Neural Cellular Automata for Modular Traffic Forecasting*. ESWA 2026. https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6408268
- Louis & Parker. *Implementing Graph Neural Cellular Automata*. Stanford CS224W blog. https://medium.com/stanford-cs224w/implementing-graph-neural-cellular-automata-098db0bca011

**Meshes / particles / other domains**
- Pajouheshgar, Xu, Mordvintsev, Niklasson, Zhang, Süsstrunk. *Mesh Neural Cellular Automata*. ACM TOG 2024. https://arxiv.org/abs/2311.02820
- Kim, Pajouheshgar, Süsstrunk, Jakob, Park. *Neural Particle Automata*. 2026. https://arxiv.org/abs/2601.16096
- *3D Texture Synthesis Using Graph Neural Cellular Automata*. KTH master's thesis 2023. https://www.diva-portal.org/smash/get/diva2:1787727/FULLTEXT01.pdf

**Grid NCA foundations & variants**
- Mordvintsev, Randazzo, Niklasson, Levin. *Growing Neural Cellular Automata*. Distill 2020. https://distill.pub/2020/growing-ca
- Niklasson, Mordvintsev, Randazzo, Levin. *Self-Organising Textures*. Distill 2021. https://distill.pub/selforg/2021/textures
- Mordvintsev, Randazzo, Fouts. *Growing Isotropic Neural Cellular Automata*. ALIFE 2022. https://arxiv.org/abs/2205.01681
- Randazzo, Mordvintsev, Niklasson, Levin. *Growing Steerable Neural Cellular Automata*. ALIFE 2023. https://arxiv.org/abs/2302.10197
- Tesfaldet, Nowrouzezahrai, Pal. *Attention-based Neural Cellular Automata (ViTCA)*. NeurIPS 2022. https://arxiv.org/abs/2211.01233
- Ruiz, Vilalta, Moreno-Noguer. *Neural Cellular Automata Manifold*. ICLR 2021. https://arxiv.org/abs/2006.12155
- Menta et al. *Latent Neural Cellular Automata for Resource-Efficient Image Restoration*. 2024. https://arxiv.org/abs/2403.15525
- Palm, González-Duque, Sudhakaran, Risi. *Variational Neural Cellular Automata*. ICLR 2022. https://openreview.net/forum?id=7fFO4cMBx_9
- Earle et al. *Pathfinding Neural Cellular Automata*. 2023. https://openreview.net/forum?id=CU8BwVAzLme
- Spitznagel & Keuper. *A New Kind of Network? Review and Reference Implementation of Neural Cellular Automata* (NCAtorch). 2026. https://arxiv.org/abs/2604.24990
- Najarro, Sudhakaran, Glanois, Risi. *HyperNCA: Growing Developmental Networks with Neural Cellular Automata*. ICLR 2022. https://arxiv.org/abs/2204.11674
- Barylli, Béna, Mordvintsev, Nisioti, Risi. *Self-Organising Digital Circuits*. 2026. https://arxiv.org/abs/2608.02606
- Wulff & Hertz. *Learning Cellular Automaton Dynamics with Neural Networks*. NeurIPS 1992.
- Gilpin. *Cellular automata as convolutional neural networks*. Phys. Rev. E 2019. https://arxiv.org/abs/1809.02942

**GNN machinery (anisotropy / positional encodings / aggregation)**
- Beaini, Passaro, Létourneau, Hamilton, Corso, Liò. *Directional Graph Networks*. ICML 2021. https://arxiv.org/abs/2010.02863
- Dwivedi et al. *Benchmarking Graph Neural Networks*. 2023. https://arxiv.org/abs/2003.00982
- Kreuzer, Beaini, Hamilton, Létourneau, Tossou. *Rethinking Graph Transformers with Spectral Attention*. NeurIPS 2021. https://arxiv.org/abs/2106.03893
- You, Ying, Leskovec. *Position-aware Graph Neural Networks*. 2019. https://arxiv.org/abs/1906.04817
- Corso, Cavalleri, Beaini, Liò, Veličković. *Principal Neighbourhood Aggregation for Graph Nets*. NeurIPS 2020. https://arxiv.org/abs/2004.05718
- Satorras, Hoogeboom, Welling. *E(n) Equivariant Graph Neural Networks*. ICML 2021. https://arxiv.org/abs/2102.09844
