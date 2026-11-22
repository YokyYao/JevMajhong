# 304 regression and a registered conditional RI-loss control

2026-11-23, Asia/Dubai. The current hard-action-CE basic run is still active;
the608 final is the sole candidate. A different fresh class-loss control is
registered/preflighted, not trained. These preparations and matched development
results do not achieve the full four-course/v8 objective.

## Current304: worse matched games despite lower CE

GPU315754 completed33s/exit0:0 on ws-l3-004. Its read-only game audit verifies
the exact sixteen seed/profile/seat pairs, saved identity, completion, score
ledgers, ranks and win/deal-in counts. The64-match v8/old-control caches are
restricted to these same sixteen matches. No new model games are generated.

| Actual matched16 policy | Mean net points | Top two | Hand win | Deal-in |
|---|---:|---:|---:|---:|
| Current basic152 |+4,756.25|50.00%|20.91%|12.87%|
| Current basic304 |−7,181.25|31.25%|11.00%|20.07%|
| Frozen v8 |+3,612.50|62.50%|22.37%|10.34%|
| Earlier completed plain584 |−1,193.75|62.50%|20.07%|12.86%|

The304-v8 paired mean is−10,793.75 net points/match and−31.25 percentage points
of top-two. There are only two independent seed groups, so no strength claim
or confidence interval is supported. The preceding favorable152 net mean was
not a stable gain; do not select it as a replacement final candidate.

Full304 validation CE falls to1.260962, ordinary-discard agreement49.16% and
plain CE1.402584. Riichi acceptance is284/324=87.65%, correct refusal83/308=26.95%.
Legacy confusion is TP34/FN3/TN20/FP50; teacher TP250/FN37/TN63/FP175. This is
another direction shift after228's low acceptance/high refusal. The original
aggregate/legacy/teacher quality grades still fail. The next actual game point
is456, then608/final64; the existing run remains unchanged.

## A full isolated fresh control is now registered

The new registrar/trainer/independent auditor implement the
[conditional class-loss plan](../plans/curriculum-riichi-control-execution-20261123.md).
Fixed base, fresh initialization/seed, same source data/masses, LR/optimizer,
sampler, four epochs608/57,140 exposures and raw serving remain identical.
Only the RI binary class component changes; no old model is loaded as a parent.
Source objective, original action CE and components are recorded separately,
and all original source-quality grades remain intact.

Recorded315754 preflight evidence:

- 186 curriculum checks and source compilation pass, including launch-trigger
  fixture rejection for a midpoint, incomplete audit,16-game result, forged
  grade, inconsistent counts or already-qualified basic model.
- All17,551 actual TRAIN/validation materialized inputs retain every original
  field after removing private RI-loss indices. Actual trainer arguments match
  the original except output. All four sampler permutations are complete;
  the three already-recorded original epoch orders match exactly.
- The changed validation-profile routine reproduces the original common CE,
  agreement, source counts and confusion on actual saved development snapshots.
  It adds the changed objective rather than renaming it as the old CE.
- No control optimizer update or language-model parameter load occurs during
  registration/preflight. Actual Qwen initialization, all initial tensor/menu/
  game equality, real updates/full fit and final reload remain pending.

GPU315760 subsequently verifies the actual pinned text encoder on **every**
14,285 TRAIN and3,266 validation menu, before/after private loss annotation.
Packed token IDs, positions, segments, decision/option indices and labels are
identical, with no truncation. Managed entry Python compilation and Slurm shell
syntax checks pass. No language-model parameters are loaded or updated. All
previous source/active/history bindings remain unchanged.

The managed workflow is prepared to wait for the actual original608 audit and
terminal-archive task. After source CI is verified, a bounded GPU job may be
queued with those dependencies. It will launch the fresh control only after
recomputing the **completed** original source-quality failure; if all source
grades pass, it records an explicit skip with no model. Separate afterok audit
and terminal/scalar-artifact jobs either follow successful fit or record the
same skip. Invalid dependencies cancel. No login polling, detached computation,
silent partial restart, failed-parent admission or release occurs.

Only source/plans/aggregate evidence are public. New raw hands, teacher labels,
per-record predictions, derived weights and research checkpoints stay private.
Main, practical v8/Windows and the final unseen256-match-per-policy grid are
unchanged. Full independent comparison follows the complete curriculum, not
this preparation or an individual16-game point.
