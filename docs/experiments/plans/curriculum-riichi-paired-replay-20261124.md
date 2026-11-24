# Conditional 3A RI replay-order experiment

The active 2400-update 3A fit remains frozen. Its 1800-step validation shows
legacy RI discrimination declining while teacher RI discrimination improves.
Cached source/schema analysis does not establish a cause. This proposal tests
one possible optimization of replay dynamics without changing labels or
increasing the number of examples.

## What the prospective sampler changes

`tools/curriculum_riichi_pair_order_v1.py` takes an original complete epoch
permutation. Within each existing 511-record mixing block it places as many
legacy and teacher RI yes/no pairs as possible into the same 73-record
gradient accumulation. It retains each original record once, exact block
membership, all19 source counts, original update sizes and the68-record tail.
It neither oversamples nor changes source weights or model-visible inputs.

The paired order is deterministic from its separately recorded RNG. The
actual TRAIN preflight reproduced both original registered permutation hashes
before deriving the proposal. Five sampler checks passed inside approved GPU
resources, covering determinism, complete permutations, unchanged block
membership, unequal classes and the actual-size tail.

The following are **counts computed from two complete prospective epoch
orders**, not new optimizer observations or a trained-model result:

| Across 2400 updates | Original order | Proposed order |
|---|---:|---:|
| Legacy RI: updates with both labels |71|361|
| Legacy RI: updates without either label |1584|1864|
| Teacher RI: updates with both labels |706|1418|
| Teacher RI: updates without either label |476|666|

Both schedules contain the same87,595 originals per epoch and175,190 total
exposures. Legacy RI has388 positive/544 negative exposures; teacher RI has
2150/1624. The proposal concentrates opposite-label replay into more shared
updates but makes source-free updates more frequent. It does **not** make
every update see RI, increase source coverage or prove lower gradient variance.
Loss mass also is not measured gradient magnitude.

## When to run it

First finish the active original fit, final64 development games and independent
audit. If its original criteria pass, continue into queued3B; this prototype
is not a reason to interrupt successful advancement. If RI retention fails,
preserve that grade and inspect other failed conditions before admitting a
repair experiment. Pairing alone is not an explanation for call or native
performance failures.

For a controlled retry, start from the **same qualified calls388 parent** with
a fresh optimizer and the original two-epoch data, weights, loss, learning-rate
schedule and policy. Change only the within-block pairing/order RNG, freeze
both new permutations and use a fresh run directory. This is a new3A attempt,
not continuation of the active3A parameters and not a rewrite of its history.
Its parameter ancestry would remain basic608→calls388→new3A2400.

The conditional entry point is now implemented in
`tools/curriculum_progression_paired_v1.py`. Its registrar rejects an
unfinished control, a successful original grade, or a failed experiment
without an actual RI failure. It retains the original failed grade, verifies
the successful original fit terminal and independent audit, and seals a fresh
registration only after confirming the same qualified calls388 initializer.
The active3A checkpoint is a control, never the initializer of this retry.

The trainer delegates to the original3A driver and changes its epoch-order
RNG factory only. It retains the original tensor/menu/game warm-start checks,
loss components, checkpoint fence, native evaluator and independent final
auditor. A separate strict comparison rejects changes to base/parent,
numerical settings, learning rate, seed, exposure or original advancement
criteria. Control and retry output directories are isolated.

Five additional entry-point checks passed. The real TRAIN driver preflight
reproduced both complete original and proposed order hashes, reconciled all
2400 prospective update-count records, preserved baseline/global RNG states,
and rejected duplicate candidate RNGs, an extra epoch and a corrupt expected
order before mutation. The bounded shell template passed syntax checking.
These are sampler/entry-point checks; they do not exercise actual model
warm-start, registration against a completed original control, or a model fit.
The original fit is still running, so the retry has **not registered or started**.

After the original final/audit actually qualifies this conditional experiment,
the bounded GPU template can be submitted from the repository root (create
`logs/` first, and use the cluster's permitted node/partition settings):

```bash
task_fit_id=$(sbatch --parsable --export=ALL,JEV_PAIRED_PHASE=run \
  cluster/curriculum-progression-paired-v1.slurm)
sbatch --parsable --dependency="afterok:${task_fit_id}" --kill-on-invalid-dep=yes \
  --time=02:00:00 \
  --export="ALL,JEV_PAIRED_PHASE=audit,JEV_PAIRED_FIT_JOB=${task_fit_id}" \
  cluster/curriculum-progression-paired-v1.slurm
```

Registration refuses an existing retry directory; it never overwrites a run.
Fit and audit have finite GPU allocations, logs, subprocess deadlines and exit
receipts. The template does not rewire existing3B jobs. A subsequently qualified
retry would need its own explicit next-course parent seal/registration.

Keep the original2400 final
selection, full validation, common16 development curves, final64 comparison
and original advancement thresholds. Compare its outcome with the completed
original fit, both parent execution routes and frozen v8; retain all failed
conditions and do not select an intermediate noisy peak. Validation cases do
not enter construction of the new TRAIN order. No final-test seeds open here.

All calculation runs inside bounded scheduler GPU resources. Publish source,
plan and aggregate counts only; per-record orders, data, weights and detailed
diagnostics remain private. Production and Windows models are unchanged.
