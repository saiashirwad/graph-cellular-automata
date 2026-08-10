# GNN Operators for an NCA-Style Iterated Local Rule

Research notes: survey of message-passing / graph-convolution operators, with a
focus on what is worth bolting onto a Graph Neural Cellular Automaton whose
perception step is currently

```
percept_v = concat[ x_v,  mean_{u in N(v)} x_u,  mean_{u in N(v)} (x_u - x_v) ]
update:    x_v <- x_v + gate( MLP(percept_v) )
```

A single fixed aggregation (plain mean), no edge weights, no attention, no
multi-hop, one shared MLP applied at every node and every step.

---

## 0. What the current rule already is

Before surveying the literature, two observations that frame everything below.

1. **The percept is a reaction-diffusion engine.** `mean(x_u) - x_v` is the
   (degree-row-normalized) graph Laplacian: `(D^{-1}A x)_v - x_v = -(D^{-1}L x)_v`.
   A rule of the form `MLP(x, mean-neighbor, Laplacian)` is precisely the
   structure of a reaction-diffusion system, which is the classical mechanism
   for self-organizing pattern formation (Turing patterns). The grid NCA of
   Mordvintsev et al. (2020) uses Sobel/gradient kernels for the same reason:
   the difference term is the spatial derivative term. So the current rule is
   already aligned with what NCA dynamics need; the interesting questions are
   what it *cannot* express (counting), what it cannot *select* (per-edge
   weights), and how fast it *propagates* (one hop per step).

2. **The third term is linearly redundant.** `mean(x_u - x_v) = mean(x_u) - x_v`,
   which lies in the span of the other two inputs. As an MLP input it adds no
   new function class; it only re-bases the coordinates (exposes the Laplacian
   direction explicitly, like a residual connection). Keeping it is harmless
   and probably good for conditioning; dropping it loses nothing in theory.

---

## 1. The classic taxonomy

### GCN — symmetric-normalized spectral convolution
**Kipf & Welling, "Semi-Supervised Classification with Graph Convolutional Networks", ICLR 2017.**
Aggregation: `h'_v = sigma( W * sum_{u in N(v) u {v}} h_u / sqrt(d_v d_u) )`.
This is the first-order truncation of a spectral (Chebyshev) filter with
lambda_max ~ 2, plus a self-loop.
**What it adds over plain mean:** the *symmetric* degree normalization
`1/sqrt(d_v d_u)`. Compared to plain mean (`1/d_v`), it down-weights
contributions from high-degree neighbors and gives the aggregation a bounded,
well-behaved spectrum (eigenvalues in [-1, 1]) — important for anything that
is iterated. GCN is the reason "which normalization" is a design decision, not
a detail.

### ChebNet — polynomial (multi-hop) filters
**Defferrard, Bresson & Vandergheynst, "Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering", NeurIPS 2016.**
Aggregation: `sum_{k=0..K} theta_k T_k(L~) x`, a degree-K polynomial of the
(scaled) Laplacian, giving a K-hop localized filter with learnable per-band
coefficients.
**What it adds over plain mean:** a *multi-radius receptive field in one
step* (2-hop, 3-hop terms) and the ability to weight different "frequency
bands" (smooth vs oscillatory components) of the neighborhood. The K=1 case
with the right coefficients reduces to mean-plus-Laplacian — i.e., the current
NCA percept is already a cheap K=1 Cheb filter.

### GraphSAGE — the concat(own, aggregate) pattern and learned aggregators
**Hamilton, Ying & Leskovec, "Inductive Representation Learning on Large Graphs", NeurIPS 2017.**
Aggregation: `h'_v = sigma( W * [ h_v || agg({h_u}) ] )` with three choices of
`agg`: mean, LSTM (over a random neighbor ordering), and pooling (elementwise
max over `MLP(h_u)` messages).
**What it adds over plain mean:** (a) it canonized the
`concat(own, aggregate)` pattern that the NCA percept already uses; (b) the
*pooling aggregator* applies a nonlinear transform to each neighbor before
aggregating — an elementwise-max over a shared MLP is permutation-invariant,
parameter-shared, and strictly more expressive than mean on raw states; (c)
the LSTM aggregator is the cautionary tale: it needs a neighbor ordering and
is expensive, and it is not permutation invariant.

### GAT — attention-weighted aggregation
**Velickovic, Cucurull, Casanova, Romero, Lio & Bengio, "Graph Attention Networks", ICLR 2018.**
Aggregation: `h'_v = sigma( sum_u alpha_uv W h_u )` with
`alpha_uv = softmax_u( a^T [W h_v || W h_u] )`, a shared scoring function over
edges; multi-head attention optional.
**What it adds over plain mean:** *data-dependent, learned per-edge weights*
— the model can choose which neighbors matter (useful when one strong signal
sits among many weak ones), and it is fully inductive (no reliance on
precomputed graph structure). Caveat: because softmax normalizes, the
aggregation is a convex combination — a *state-dependent smoothing operator* —
which interacts with over-smoothing in deep/iterated use. Also note **GATv2**
(Brody, Alon & Yahav, "How Attentive are Graph Attention Networks?", ICLR
2022): GAT's attention scores are "static" (the ranking of neighbors is
independent of the query node, a strong limitation); GATv2 fixes the score
order and is the version to use if attention is adopted.

### GIN — sum aggregation for maximal expressive power
**Xu, Hu, Leskovec & Jegelka, "How Powerful are Graph Neural Networks?", ICLR 2019.**
Aggregation: `h'_v = MLP( (1+eps) h_v + sum_{u in N(v)} h_u )`.
**What it adds over plain mean:** *sum over neighbors is injective over
multisets* (given an injective update), which makes the architecture provably
as expressive as the 1-dimensional Weisfeiler-Lehman test — the strongest
possible among message-passing GNNs. The paper also proves that mean and max
aggregators *cannot* distinguish certain multisets (details in Section 2).
Practical caveat: on irregular graphs, raw sums scale with degree; the paper
uses batch normalization, and in an iterated rule an unnormalized sum risks
divergence (spectral radius grows with max degree).

### APPNP / diffusion models — fixed linear propagation with a teleport
**Gasteiger, Bojchevski & Gunnemann, "Predict then Propagate: Graph Neural Networks meet Personalized PageRank", ICLR 2019.**
Aggregation: run the MLP once, then propagate k times with the personalized
PageRank operator `Pi = alpha (I - (1-alpha) A~)^-1` (computed cheaply by
power iteration). The `alpha` "teleport" term re-injects the node's own state
at every propagation step.
**What it adds over plain mean:** *controllable receptive-field radius without
depth*, and an explicit knob (`alpha`) that interpolates between purely local
and effectively global information while provably limiting over-smoothing.
This is the cleanest way to add multi-hop information to a model whose MLP
must stay a single shared rule. GRAND (Chamberlain et al., "GRAND: Graph
Neural Diffusion", ICML 2021) and PDE-GCN (Alet et al., NeurIPS 2021) push
this to continuous diffusion/PDE views of GNNs — useful framing for an
iterated rule (an NCA step is one Euler step of a graph ODE), but their
learned-diffusivity machinery is more than the NCA needs.

### MPNN — the general framework
**Gilmer, Schoenholz, Riley, Vinyals & Dahl, "Neural Message Passing for Quantum Chemistry", ICML 2017.**
Aggregation: `m_v = sum_{u in N(v)} M(h_v, h_u, e_uv)`, then `h'_v = U(h_v, m_v)`
with arbitrary message function M and update function U.
**What it adds over plain mean:** the *vocabulary*: an NCA rule is an MPNN
with M = identity/edge-feature readout and U = gated MLP. The framework's main
openings are (a) edge features `e_uv` in the message, and (b) *learned
messages* (M as a small shared MLP) instead of raw states.

### Gated graph networks — the edge-gating idea
**Li, Tarlow, Brockschmidt & Zemel, "Gated Graph Sequence Neural Networks", ICLR 2016.**
Aggregation: `m_v = sum_u A_uv h_u` with edge-type-specific weights, then a
GRU-style gated update `h'_v = GRU(h_v, m_v)`.
**What it adds over plain mean:** per-edge (edge-type) weighting and
recurrent gating of the update. This is the prototype for "edge gating" as a
cheap alternative to attention: scalar per-edge multipliers, no softmax
normalization, still permutation-invariant and parameter-shared.

### PNA — the practical answer to "which aggregator"
**Corso, Cavalleri, Beaini, Li & Velickovic, "Principal Neighbourhood Aggregation for Graph Nets", NeurIPS 2020.**
Aggregation: all of `{mean, max, min, std}` combined with degree scalers
`{identity, log(1+d), 1/log(1+d)}`, giving 12 aggregate channels.
**What it adds over plain mean:** the observation that *no single aggregator
is enough* — different tasks need different statistics — and that degree-based
scaling interpolates between mean-like and sum-like behavior. Empirically the
most robust aggregator in benchmarking; the multi-aggregator trick is nearly
free (the extra aggregators have no parameters if applied to raw states).

### Summary table

| Operator | Aggregation | Adds over plain mean | NCA-relevant cost |
|---|---|---|---|
| GCN (Kipf & Welling 2017) | symmetric-norm sum, self-loop | bounded spectrum, hub-aware weighting | zero params |
| ChebNet (Defferrard et al. 2016) | K-hop Laplacian polynomial | multi-radius field, band weights | zero params (fixed K) |
| GraphSAGE (Hamilton et al. 2017) | mean / LSTM / max-pool + concat | concat pattern; nonlinear pooling agg | small shared MLP (pooling) |
| GAT (Velickovic et al. 2018) | softmax-weighted sum | data-dependent edge weights | per-edge scoring |
| GIN (Xu et al. 2019) | sum + injective MLP | 1-WL expressivity | needs degree control |
| APPNP (Gasteiger et al. 2019) | PPR diffusion, teleport alpha | depth-free radius, anti-over-smoothing | zero params (fixed) |
| MPNN (Gilmer et al. 2017) | general M, U | edge features, learned messages | framework |
| GGNN (Li et al. 2016) | edge-type weights + GRU gate | edge gating | per-edge scalar |
| PNA (Corso et al. 2020) | mean/max/min/std + degree scalers | complementary stats + degree info | ~zero params |

---

## 2. Expressive power: WL limits, sum vs mean vs max, and whether it matters here

### The 1-WL ceiling
Message-passing GNNs are at most as powerful as the 1-dimensional
Weisfeiler-Lehman test at distinguishing non-isomorphic graphs, and GIN
achieves exactly 1-WL (Xu et al., ICLR 2019; the k-WL hierarchy and
higher-order GNNs are covered by Morris et al., "Weisfeiler and Leman Go
Neural: Higher-Order Graph Neural Networks", AAAI 2019). The upper bound comes
from the local, multiset-structured update: nodes whose neighborhoods are
indistinguishable as multisets get identical representations, forever.

### Why sum > mean > max (in theory)
Xu et al. (2019) prove that with continuous feature vectors, mean and max
aggregators cannot distinguish certain multisets that sum can:
- **mean loses cardinality.** `mean({2}) = mean({2,2}) = 2`. A node with one
  neighbor of value 2 and a node with two such neighbors see the same percept.
  More generally, mean collapses the multiset onto its first moment, losing
  count and spread information.
- **max loses everything except the maximum.** `max({1,2}) = max({2}) = 2`;
  counts and all non-maximal entries are discarded.
- **sum is injective over multisets** (of bounded size, given injective node
  features), so a sum-aggregating GNN with an injective update can in
  principle compute any 1-WL-distinguishable function.

Two important qualifications. First, the failure modes are about *cardinality*:
mean plus a degree feature (or PNA's degree scalers) recovers what mean throws
away, since `count x mean = sum`. Second, the theory is about single-pass
expressive power with a final readout — the setting where every layer's loss
of information is permanent.

### Does it matter for an iterated dynamical system?
The NCA setting differs from one-shot prediction in three ways, and each one
changes the calculus.

1. **Stability beats injectivity when you iterate.** Iterating a *linear*
   mean-aggregation operator is a diffusion/Markov process that converges to
   the Perron vector — the over-smoothing phenomenon (Li, Han & Wu, "Deeper
   Insights into Graph Convolutional Networks for Semi-Supervised Learning",
   AAAI 2018; Oono & Suzuki, "Graph Neural Networks Exponentially Lose
   Expressive Power for Node Classification", ICLR 2020). Over-smoothing is
   exactly what an NCA must *avoid* (it would kill the pattern), and the MLP
   nonlinearity plus the Laplacian/difference term are what prevent collapse.
   The operator norm of the aggregation matters more here than its
   injectivity: plain mean (row-normalized) and symmetric-normalized
   aggregation have spectral radius <= 1 and are safe to iterate; *raw sum has
   spectral radius = max degree* and will blow up on irregular graphs. So the
   theoretical ranking sum > mean > max does **not** transfer to an iterated
   rule: mean (or normalized variants) is the stable default, and sum should
   only be used with degree normalization or degree scalers (PNA-style).

2. **The rule needs to distinguish configurations, not graphs.** The WL
   analysis is about global graph isomorphism / readout tasks. A local update
   rule instead needs to answer, at every node, "given my state and my
   neighborhood, what should change?" — and it must be able to answer
   *differently* for configurations that demand different dynamics. The sharp
   example is counting: Conway's Game of Life is literally a sum-aggregation
   rule (count of live neighbors in {0..8}). Behaviors like contact
   inhibition ("stop growing when crowded"), recruitment ("activate when at
   least k neighbors are active"), or any density-dependent rate cannot be
   expressed with mean or max alone, because the percept cannot see the
   *number* of contributing neighbors. This is the one place where the GIN
   critique bites a CA: if the target dynamics are count-sensitive, mean/max
   percepts are provably insufficient — and the cheap fix is a degree feature
   or a sum channel, not a redesign.

3. **Information propagates over time, so single-step injectivity is less
   critical — but speed matters.** Over hundreds of steps, neighbor
   information reaches a node through repeated 1-hop diffusion, so the
   percept does not need to encode the whole neighborhood at once. What does
   matter is *how many steps* it takes to establish long-range coupling: the
   mixing time / graph diameter. This is where multi-hop percepts (ChebNet,
   SIGN, APPNP) earn their keep — they shorten the effective radius per step
   at zero parameter cost if implemented as fixed linear diffusion. (The
   over-squashing bottleneck of Alon & Yahav, ICLR 2021, is mostly a
   one-shot/deep-stack concern; an NCA has no fixed readout, but the
   propagation-speed concern is the same family of problem.)

**Bottom line:** the mean-based percept is the right *stable* default for
diffusive dynamics, and the Laplacian term is the pattern-forming engine. The
real expressive gaps for a dynamical system are (a) counting/degree
information, (b) per-edge selectivity (which neighbor matters), and (c)
propagation speed. WL injectivity of the local aggregation is a nice-to-have,
not the binding constraint.

---

## 3. Recommendations: what to bolt on, what to skip

### Tier 1 — cheap, zero (or near-zero) parameters, high value

1. **Degree features.** Add `log(1 + d_v)` (and/or `1/d_v`) to the percept.
   Zero parameters; restores the count information that mean aggregation
   provably discards (Section 2). This is the single cheapest fix for the
   main theoretical gap, and it is well-supported empirically (degree scalers
   in PNA, degree features in GIN-style models). Test whether count-sensitive
   behaviors (growth stopping, density-dependent rates) improve.

2. **Multi-aggregator percept (PNA-lite).** Concat `[mean, max, std]` of
   neighbor states (optionally `min`). Zero parameters if applied to raw
   states. Max captures "is there a strong signal anywhere", std captures
   spread — complementary statistics that mean alone never provides. If
   budget allows, upgrade max to the GraphSAGE pooling aggregator (elementwise
   max over a small shared MLP on neighbor states): a few extra parameters,
   still permutation-invariant and parameter-shared. This is the highest
   value-per-effort expressive upgrade.

3. **Normalization audit.** You currently use `D^{-1}A` (plain mean). Try
   symmetric GCN normalization `D^{-1/2} A D^{-1/2}` (with a self-loop) — zero
   parameters, changes the spectrum (bounds eigenvalues, down-weights
   high-degree hubs) and therefore changes stability and dynamics on irregular
   graphs. Also decide deliberately whether the rule should be
   degree-sensitive (sum-like) or degree-invariant (mean-like), and verify the
   chosen rule transfers across graphs of different densities. For an iterated
   rule, keep the spectral radius of the linear part comfortably below 1.

4. **Multi-hop percept via precomputed powers (SIGN-style) or PPR (APPNP-style).**
   Add one extra channel: the mean over the 2-hop frontier (`A^2 x`, each hop
   degree-normalized), or an APPNP teleport mix `alpha * x + (1-alpha) * Ax`
   iterated k times at *precomputation time*. Zero learned parameters if the
   diffusion is fixed. This directly addresses propagation speed (idea 3 of
   Section 2) and is the NCA-friendly fragment of ChebNet/APPNP — the MLP
   stays a single shared rule; only the perception gains radius. Start with
   one extra hop; more hops per step effectively raise the CA's radius and can
   destabilize the discrete-time dynamics (higher-order diffusion), so keep
   coefficients small and check the rollout for divergence.

5. **Keep the difference term, and know why it is there.** It is the
   Laplacian (reaction-diffusion) direction — the pattern-forming engine, and
   the grid-NCA analog of Sobel filters. It is linearly redundant with
   `[x, mean]` but likely good for conditioning. Optionally replace it with
   the *degree-weighted* normalized Laplacian to make the gradient estimate
   principled on irregular graphs. Zero cost either way.

### Tier 2 — moderate cost, plausible, worth an ablation

6. **Edge gating (GGNN-style).** A per-edge scalar gate
   `g_uv = sigmoid(MLP([h_u, h_v, e_uv]))` multiplying the message before
   aggregation. Parameter-shared, permutation-invariant, no softmax — so it
   does not force the aggregation to be a convex combination (it preserves
   counting better than attention, since gates can all be near 1). Cost: O(E)
   per step, plus one small MLP. This is the cheapest way to make the
   aggregation *selective*.

7. **GATv2-style local attention.** There is direct precedent: Tesfaldet,
   Nowrouzezahrai & Pal, "Attention-based Neural Cellular Automata", NeurIPS
   2022, replaced the grid-NCA update rule with locally-constrained
   self-attention and got competitive growth results — so attention in an NCA
   rule is known to train. On graphs, use GATv2 scoring (Brody et al. 2022;
   plain GAT has the static-attention flaw). Caveats: softmax makes
   aggregation a state-dependent smoothing operator, so long rollouts risk
   over-smoothing; per-edge scoring adds compute and per-step Jacobian
   variance, so training may need the standard NCA tricks (short BPTT
   windows, stochastic update masks). Only worth it if the dynamics need
   "attend to the most informative neighbor" — e.g., one strong signal among
   many weak ones.

8. **GIN-style sum, but only with degree control.** Raw sum is expressive but
   unstable on irregular graphs (spectral radius = max degree). If
   degree-sensitivity is genuinely wanted, use sum *with* PNA-style degree
   scalers or a learned per-channel normalization, not bare sum.

### Overkill / fights the NCA paradigm

- **Global attention / Graph Transformers.** O(n^2), non-local, breaks the
  shared-local-rule structure that makes an NCA an NCA. If long-range coupling
  is the problem, multi-hop diffusion (Tier 1) or simply more steps is the
  natural lever.
- **k-WL / higher-order GNNs** (Morris et al. 2019). Combinatorial blowup,
  designed for global isomorphism power; a local update rule does not need it.
- **GraphSAGE LSTM aggregator.** Requires a neighbor ordering — breaks
  permutation invariance and adds cost for a setting (irregular graph
  neighborhoods) where ordering is meaningless.
- **Full spectral convolution (eigen-decomposition).** Graph-specific, not
  parameter-shared, not local; its useful content is captured by K=1/2
  polynomial filters.
- **Stacking many GCN layers per step to gain range.** The NCA already
  iterates; depth per step adds parameters without adding long-range
  capability beyond what additional steps provide. If the rule is too
  expensive or too deep, prefer a wider percept over a deeper rule.
- **APPNP's full "predict-then-propagate" decoupling.** In a CA the MLP must
  see the state every step (the rule *is* the dynamics); you cannot
  precompute the prediction. Borrow the PPR diffusion as a percept component
  (Tier 1 #4) and skip the rest.

### Suggested experiment matrix

1. Baseline: `[x, mean, mean-diff]`.
2. `+ deg`: add `log(1+d)`.
3. `+ multi-agg`: `[x, mean, max, std, mean-diff]`.
4. `+ 2-hop`: add mean over 2-hop frontier (fixed, normalized).
5. `+ gate`: edge-gated messages (Tier 2 #6).
6. `+ GATv2`: local attention (Tier 2 #7).

Diagnostics worth tracking, because they are the dynamical-system version of
"expressivity": per-step spectral radius / contraction of the linearized rule,
feature-rank collapse over a rollout (over-smoothing index à la Oono &
Suzuki), and transfer of the trained rule across graphs with different degree
distributions (degree-invariance vs degree-sensitivity of the rule).

---

## References

- Kipf & Welling. *Semi-Supervised Classification with Graph Convolutional Networks.* ICLR 2017. arXiv:1609.02907.
- Defferrard, Bresson & Vandergheynst. *Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering.* NeurIPS 2016. arXiv:1606.09375.
- Hamilton, Ying & Leskovec. *Inductive Representation Learning on Large Graphs.* NeurIPS 2017. arXiv:1706.02216.
- Velickovic, Cucurull, Casanova, Romero, Lio & Bengio. *Graph Attention Networks.* ICLR 2018. arXiv:1710.10903.
- Brody, Alon & Yahav. *How Attentive are Graph Attention Networks?* ICLR 2022. arXiv:2105.14491.
- Xu, Hu, Leskovec & Jegelka. *How Powerful are Graph Neural Networks?* ICLR 2019. arXiv:1810.00826.
- Morris, Ritzert, Fey, Hamilton, Lenssen, Rattan & Grohe. *Weisfeiler and Leman Go Neural: Higher-Order Graph Neural Networks.* AAAI 2019. arXiv:1810.02244.
- Gasteiger, Bojchevski & Gunnemann. *Predict then Propagate: Graph Neural Networks meet Personalized PageRank.* ICLR 2019. arXiv:1810.05997.
- Gilmer, Schoenholz, Riley, Vinyals & Dahl. *Neural Message Passing for Quantum Chemistry.* ICML 2017. arXiv:1704.01212.
- Li, Tarlow, Brockschmidt & Zemel. *Gated Graph Sequence Neural Networks.* ICLR 2016. arXiv:1511.05493.
- Corso, Cavalleri, Beaini, Li & Velickovic. *Principal Neighbourhood Aggregation for Graph Nets.* NeurIPS 2020. arXiv:2004.05718.
- Rossi, Frasca, Chamberlain, Eynard, Bronstein & Monti. *SIGN: Scalable Inception Graph Neural Networks.* arXiv:2004.11198 (2020).
- Li, Han & Wu. *Deeper Insights into Graph Convolutional Networks for Semi-Supervised Learning.* AAAI 2018. arXiv:1801.07606.
- Oono & Suzuki. *Graph Neural Networks Exponentially Lose Expressive Power for Node Classification.* ICLR 2020. arXiv:1905.10947.
- Alon & Yahav. *On the Bottleneck of Graph Neural Networks and its Practical Implications.* ICLR 2021. arXiv:2006.05205.
- Chamberlain, Rowbottom, Gorinova, Bronstein, Webb & Rossi. *GRAND: Graph Neural Diffusion.* ICML 2021. arXiv:2106.10934.
- Alet, Jeewajee, Bauza Villalonga, Rodriguez, Lozano-Perez & Kaelbling. *PDE-GCN: Novel Architectures for Graph Neural Networks Motivated by Partial Differential Equations.* NeurIPS 2021. arXiv:2108.01938.
- Mordvintsev, Randazzo, Niklasson & Levin. *Growing Neural Cellular Automata.* Distill, 2020.
- Grattarola, Livi & Alippi. *Learning Graph Cellular Automata.* NeurIPS 2021. arXiv:2110.14237.
- Tesfaldet, Nowrouzezahrai & Pal. *Attention-based Neural Cellular Automata.* NeurIPS 2022. arXiv:2211.01233.
