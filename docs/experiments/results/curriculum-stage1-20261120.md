# Curriculum v1 / Stage1: fresh-base discard and riichi

Status: completed on2026-11-20 (Asia/Dubai). Main GPU job311052 and evidence audit311243 both exited0:0. The fixed1,162 updates, final validation,64 development matches, matched-grid figures and saved-model reload checks are complete. This is a completed stage1 experiment, not a released replacement for v8.

## Registered recipe

This is a newly organized curriculum based on v8 experience, not a literal replay of v8's historical parameter lineage. Qwen3.5-0.8B revision `2fc06364715b967f1860aea9cf38778875588b17` initializes a fresh rank16 LoRA and 256-dimensional action head. The pretrained backbone is frozen; no v8 or Mortal parameters initialize the student. V8/Jev Play v2 is a separate frozen comparison policy.

The public source dataset is fixed to repository commit `7ee572904bfd30387d2451a2dd1d770f90ffebf7`. The new group-isolated, context-complete stage1 split has 9,292 training and 1,972 validation decisions. Training contains 8,826 plain-discard, 194 teacher-riichi and 272 teacher-decline decisions; validation contains 1,865/37/70. Reserved source groups are excluded from all later-stage training, without using labels or rewards to choose the groups. The new split is unseen by this fresh student; some held-out groups were seen by historical v8 and cannot be presented as unseen supervised validation for v8.

The fixed recipe is two epochs, 1,162 optimizer updates, microbatch1 with accumulation16, BF16, LoRA/head peak learning rates 2e-5/1e-4, rank16, dropout0.05, AdamW weight decay0.001 and OneCycle scheduling. Plain/riichi windows receive 80%/20% loss mass; the riichi category retains the natural training accept/decline ratio. Data are not duplicated to inflate counts. The final checkpoint is preregistered; validation-minimum CE and the best noisy game point do not silently replace it.

See the [configuration](../../../configs/training/curriculum-stage1-v1.json), [reproduction commands](../../../reproduction/README.md), [data audit](curriculum-data-shape-audit-20261120.md) and [four-stage plan](../plans/curriculum-reproduction-20261120.md).

## Validation: improvement and an unresolved riichi bias

Loss below uses the fixed validation-stratum weights. Ordinary-discard accuracy is exact agreement with the observed teacher, not a win probability. Riichi acceptance is measured on the 37 teacher-accept windows; correct decline is action-type agreement on the 70 teacher-decline windows. Decline accuracy differs from exactly matching the subsequent discarded tile.

| Update | Weighted validation CE | Plain exact match | Teacher-riichi accepted | Teacher-decline correctly declined |
|---:|---:|---:|---:|---:|
|0|2.473897|15.07%|3/37|69/70|
|128|2.223563|33.35%|37/37|0/70|
|256|1.595806|44.56%|0/37|70/70|
|384|1.443278|45.58%|22/37|45/70|
|512|1.493795|47.99%|1/37|67/70|
|640|1.365277|48.79%|24/37|21/70|
|768|1.427539|46.86%|9/37|53/70|
|896|1.347942|49.92%|19/37|28/70|
|1024|1.340156|50.08%|25/37|18/70|
|1152|1.330187|50.24%|26/37|16/70|
|1162 (registered final)|1.329124|50.46%|27/37|16/70|

Numerical stability and decision stability are different. Loss and ordinary-discard agreement improve, but riichi decisions oscillate between over-accepting and over-declining. At the final checkpoint, 54/70 teacher-decline windows are incorrectly accepted; only16/70 are correctly declined, despite27/37 teacher-riichi choices being accepted. Aggregate CE alone would conceal this. Teacher choices remain imitation targets rather than proven optimal decisions. A causal explanation for the oscillation requires a separately registered follow-up; the active recipe was not retuned mid-run.

## Complete East-match development diagnostics

The rule is four-player red East, using the pinned RiichiEnv mjsoul configuration. Each periodic point uses the same two seed groups, all four controlled seats and two opponent profiles: three Efficiency opponents, or one Calling opponent with two Efficiency opponents. V8 uses the same grid. A rate is the mean of per-match rates; net points are terminal raw points minus 25,000, and observed reward is net points/8,000. These games do not provide gradients.

| Update | Matches | Hand-win rate | Dealin rate | First | Top-two | Fourth | Net points / match |
|---:|---:|---:|---:|---:|---:|---:|---:|
|0|16|6.16%|32.47%|0.00%|6.25%|68.75%|-11,437.50|
|256|16|12.57%|19.57%|12.50%|25.00%|37.50%|-8,312.50|
|512|16|27.69%|17.71%|37.50%|37.50%|37.50%|-3,331.25|
|768|16|12.60%|4.13%|18.75%|43.75%|31.25%|-800.00|
|1024|16|20.44%|25.53%|31.25%|37.50%|43.75%|-4,762.50|
|1162, matching curve subset|16|19.54%|26.18%|31.25%|37.50%|50.00%|-5,343.75|
|Frozen v8, matching subset|16|22.37%|10.34%|62.50%|62.50%|31.25%|+3,612.50|

Only two seed groups underlie each periodic point. These are development diagnostics, not precise strength estimates or an independent test. Higher hand-win rate at 1024 than 768 does not imply higher net points: dealins and terminal placement also matter. No checkpoint is selected from this table.

The final student used64 matches across eight seed groups, paired to the already completed v8 reference. Curves use only its matching16-game subset, keeping sample composition consistent across points. The full64-match result is reported below; new final-test seeds are reserved for later whole-curriculum confirmation. The64-match mean is not spliced into the16-match curve.

| Policy / same64-match grid | Hand-win | Dealin | First | Top-two | Fourth | Net points / match |
|---|---:|---:|---:|---:|---:|---:|
| Stage1 final1162 |23.93%|18.69%|32.81%|51.56%|31.25%|+18.75|
| Frozen v8 / Jev Play v2 |24.32%|10.44%|42.19%|59.38%|14.06%|+4,290.63|

The student trails v8 by7.8125 percentage points in top-two rate and4,271.875 raw points/match on this development grid. Its hand-win rate is close to v8, but its dealin and fourth-place rates are worse. This is a useful diagnostic, not proof that a particular training component caused the difference. Stage1 is a basic course, and has not reproduced v8's later call/attack/risk training. Development scores must not be promoted to independent confirmation.

Pairing whole seed groups gives approximate95% t intervals (eight clusters): student-minus-v8 top-two change−7.8125pp [−22.5253,+6.9003pp], net-points change−4,271.875 [−8,917.066,+373.316]. Both intervals cross zero; the observed means trail v8, but this small inspected development grid does not establish a precise population difference. These are not64 independent randomized match samples or an independent final test.

## Stability and evidence boundaries

The completed audit verifies all1,162 actual updates have continuous finite loss/gradient/learning-rate records, and independently reconstructs their source-weighted losses. Total exposure is17,652 plain +388 teacher-riichi +544 teacher-decline =18,584, exactly two epochs of the registered data. The first epoch ends at581. Final optimizer state contains376 parameter states all at1162, epoch2, records18,584 and the complete shuffled training order. Reserved/validation groups remain excluded from all prepared stage training sets, and every frozen data/source/config hash is unchanged.

The ordinary saved-model loader reproduces step0 on139 menus within4.04e-8 probability error and final1162 on all1,972 validation menus within1.29e-7. Final checkpoint identity is `df349cf925d7b2b85d5cb4b75be47f75ea4c821ba7add8b32ca93dca8bccceeb`. All saved validation points were recomputed from their prediction records. The audit rechecks208 cached complete-game records (v8 reference64 + five periodic student16 grids + final64), their score/rank ledgers, summaries and the matching16 subset. It plays no new games. These checks establish actual parameter updates, usable serialization and internally consistent evidence, not playing strength.

Supervised training reward is null. Reward curves show only observed complete-game outcomes. No additional games are generated for plotting. The loss, reward and rate figures read real JSONL, retain unsmoothed observations and export CSV plus standalone PNG/PDF. `plot-data-provenance.json` records the aligned sample transformation. Raw hands/events, hidden worlds and new weights remain private.

Stage1 has no win/pass supervision and does not teach yaku or han/fu scoring; legal wins are taken by the existing rule path. Calls are not the current course's supervised target. The student can still encounter call windows during complete matches, so the stage1 match result is neither a complete four-stage curriculum result nor a claim that later call/risk/placement skills have been reproduced. Game-ledger checks use the same engine; they are not an independent han/fu implementation.

The main job and its dependent audit have scheduler time limits, logs and exit receipts. Computation runs only inside GPU allocations. No automatic stage2 training or Windows model replacement is part of this fit.

## Implications for the next registered experiment

This run establishes a complete, logged fresh-base training pass and improved basic imitation, not a stable replacement for v8. The next recipe should explicitly retain riichi-accept/decline diagnostics and enough independent source groups. Sparse riichi windows combined with large per-example weights and small effective batches are a plausible source of noisy updates, but this run does not isolate their causal contribution. A follow-up can compare stratified batch construction and a gentler learning schedule against this frozen baseline, preserving the held-out groups, loss definition and game grid. Simply extending training because aggregate CE falls would not address the observed action bias.

The already prepared call/post-call course and later risk course remain separate fits requiring their own registered transfer checkpoint and replay mix. The final64 scores are not a reason to silently cherry-pick768 or to skip later courses. Keep v8 frozen for practical use while the new curriculum is evaluated.
