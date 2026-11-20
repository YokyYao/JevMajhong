# Recovering stage3 simulation-label provenance

2026-11-21, Asia/Dubai. The stage2 calling course remains active; this is a TRAIN-data audit for the next course, with no optimizer update, label change or practical-model change. It supplements the earlier [stage2/3 preparation report](curriculum-stage2-stage3-data-20261121.md), whose public-record-only audit could not reconstruct rollout uncertainty.

## Exact recovery from the private source

The public dataset anonymizes record IDs and whole-game groups. Direct ID matching therefore finds zero matches. Matching the complete state, action question and target instead recovers a unique original private row for **all4,083** simulated records, including the177 singleton-support rows previously quarantined. No two rows have the same complete-content fingerprint. The recovered498 native seeds do not overlap the reserved final-test seeds2026300000–2026300031.

Of the original3,783 TRAIN and300 validation simulated rows,925 original TRAIN rows now belong to the stricter curriculum validation partition. They remain excluded from training. The current informative TRAIN subset contains2,722 rows in339 whole-game groups, all from original TRAIN. The audit does not move any row between partitions.

For those2,722 current TRAIN rows, the new audit reads the hash-bound source manifests and395 original rollout shards; verifies original source priorities, normalized public inputs, complete eligible alternatives, shared paired-world seeds and identical control continuations; independently recomputes win, speed and dealin outcomes from settled hand logs; and reproduces the stored target probabilities. **Every target reproduces**, with maximum absolute error **2.7755575615628914e-16**. Validation rollout outcomes are not used for this reconstruction or recipe selection.

The published code requires these original private artifacts for complete reconstruction. The already public, hash-pinned dataset remains sufficient for fitting the unchanged targets. This report does not claim that the private per-world evidence is publicly downloadable.

## What the targets mean

The target reward is:

`own_win + 0.25 × 0.98^(future draws until own win) − 1.5 × dealin`

Speed is zero without an own win. Another player's win without our dealing in gives us zero utility. Net points have zero coefficient. Each intervention is compared with its common baseline continuation in the same worlds; targets average256 paired bootstrap distributions at temperature0.05, then preserve the original served support. An unobserved legal action remains masked, not a negative example. The256 bootstrap replicates reuse the stored worlds and do not create256 independent simulations.

There are2,492 records with16 worlds and230 with32. The following diagnostic is conditional on those worlds and their continuation policy:

| Phase | TRAIN records | Target argmax is baseline | Median bootstrap best-action mass | Nonbaseline selected action with positive nominal bootstrap lower bound |
|---|---:|---:|---:|---:|
| Closed discard |717|180|0.5000|60|
| Open discard |254|99|0.6618|30|
| Call window |1,503|786|0.8066|111|
| Kan window |248|98|0.6602|24|

Best-action mass splits ties rather than assigning full confidence to every tied action. The final column is a descriptive selected-action interval; it is neither a corrected post-selection confidence guarantee nor a count of expert-approved decisions. The baseline's zero self-difference is never classified as a positive improvement. Ambiguous cases remain soft targets; no confidence-based hard relabeling or automatic filtering occurred.

## Consequences for the next course

These artifacts support reusing the existing labels with their actual limits. In the inspected, hash-bound producer, opponent concealed hands stay fixed while the declared unseen-wall scope is reshuffled. The old continuation uses Discard/Call policies, and the rollout stops after the intervention hand settles. The labels are therefore conditional rollouts. They are not fully sampled beliefs about unknown hands, online self-play, expert optimality, or final-match placement labels.

Keep the proposed75% expert /25% simulation loss mass as a bounded starting proposal, with replay of ordinary decisions, riichi, calling and post-call discards. Preserve masked losses and the existing uncertain distributions. The closed-discard ambiguity argues against turning every simulated preference into a hard answer or assuming additional epochs alone repair the labels. Any targeted new worlds must have separately frozen public-information sampling, continuation policies, TRAIN-only selection and whole-group exclusions. Stage4 needs full-match outcomes and cannot inherit this hand-only reward as a placement label.

The stage3 parent, optimizer budget, mixture implementation and candidate rule still require separate registration after the actual stage2 result and audit. The present audit does not admit a parent or launch stage3. All four courses and the reserved final comparison against v8 remain required for the active goal.

## Reproduction

Inside an approved scheduler GPU allocation, with the original private artifacts present:

```bash
python tools/trace_curriculum_mc_provenance.py \
  --data-ready runs/curriculum-optimization-v2/stage3-data/data-ready.json \
  --out runs/curriculum-optimization-v2/stage3-source-trace-v2-private.json \
  --probe-shards 3
python tools/audit_curriculum_mc_worlds.py \
  --data-ready runs/curriculum-optimization-v2/stage3-data/data-ready.json \
  --out runs/curriculum-optimization-v2/stage3-world-audit-v1
python -m unittest tests.core.test_curriculum_simulation_provenance -v
```

Outputs refuse overwrite. Three new checks passed in the actual allocated GPU step: anonymous metadata cannot alter full-content identity; independent outcome accounting separates another player's win from dealing in and counts all future draws; unknown alternatives remain masked, equal outcomes share confidence, changed targets fail reproduction and held rows are rejected. The actual audit completed on GPU node`ws-l1-011`, allocation312081, with a12-minute foreground time limit. Original observations, world logs, source links and per-record confidence remain private; only the code, tests and this aggregate report are published.
