# Experiment history

The lab notebook for the perception work. Each rung of the ladder changes ONE
thing about what a node can see, and is judged against the same fixed probes.
Metrics live in trackio (`trackio show --project gnca`); this file is the
written record, kept so the story can become a blog post.

## The setup

Model: shared MLP rule on a kNN graph (`src/gnca/model.py`). Perception started
as `[x, mean(neighbors), mean(neighbors − own)]` — isotropic, count-blind, and
diffusion-biased (measured: pure `mean_diff` destroys ~95% of Dirichlet energy
in one step on our 1024-node k=8 graph).

Target: Stanford bunny surface cloud, 1536 nodes, k=12. Training: growing-NCA
recipe + damage, 8000 steps unless noted. Probes (every 1000 steps, fixed
seed/damage): `probe_grown` (MSE after growing from seed), `probe_healed` (MSE
after punching a 25% ball and rehealing), `dirich_*` (Dirichlet energy along
the rollout — how much structure the state carries; collapse = over-smoothing).

## The ladder (issues #17–#20)

1. **#20 instrumentation** — per-run tracking: run names, per-run checkpoints,
   probe Dirichlet curves, steps/sec, rollout video on each run. Done.
2. **#17 degree feature** — add `log(1+deg)` to the percept. Done, see below.
3. **#18 diffusion gate** — Perona–Malik edge gating on the Laplacian term.
4. **#19 directional perception** — MeshNCA-style geometry from node positions.

Rule of the ladder: warm-start each rung from the previous best for the fast
signal (`--init-from`, zero-padded so the rule starts functionally identical);
a from-scratch run is required before any number is quoted as final.

## Results

### Rung 0 — baseline (2026-08-10)

`--run baseline`, from scratch, 8000 steps.
Final: probe_grown 0.0459, probe_healed 0.0179, dirich_healed 0.415.

### Rung 1 — degree feature (#17) (2026-08-11)

`--run deg --init-from runs/checkpoint_bunny--baseline.pt`, 3000 steps.
Final: probe_grown 0.0369, probe_healed 0.0115, dirich_healed 0.484.

Verdict: WIN on all three metrics. Healed error −36%; the healed pattern comes
back sharper (higher sustained Dirichlet energy). Interpretation: neighbor
count — which mean aggregation provably erases — is real signal for knowing
when to stop growing. Caveat: warm-started; from-scratch confirmation pending.

## Open threads

- From-scratch `deg` run for the honest #17 number.
- #18, #19 drafts.
- Regenerate `web/umap.js` per final model + redeploy the demo.
