# Attack-course replay, loss and continuation preparation

2026-11-21, Asia/Dubai. Stage2 GPU312081 is running its registered324-update budget. The work here prepares stage3 data and its parent-verification bridge. It does not admit a stage3 parent, load a new student or run an optimizer. Stage1/2 registered inputs and their audit code remain frozen.

## Current stage2 evidence

The first full validation at81 updates uses the same4,721 records as step0. Weighted CE changes from1.116720 to1.023091. These values are comparable within this calls course; they are not comparable to the differently weighted stage1 loss.

| Metric | Step0 | Step81 |
|---|---:|---:|
| Ordinary exact action agreement |50.72%|50.35%|
| Closed riichi teacher-accept recall |70.27%|94.59%|
| Closed riichi teacher-decline recall |42.86%|14.29%|
| Chi balanced operation agreement |58.42%|72.18%|
| Pon balanced operation agreement |52.43%|75.67%|
| Ankan balanced operation agreement |52.69%|58.25%|
| Daiminkan balanced operation agreement |50.00%|58.08%|
| Kakan balanced operation agreement |50.32%|53.36%|

The mean five-call balanced gain is about10.74 percentage points. This is observed replay agreement, not proof that calling more is optimal. Riichi is increasingly biased toward acceptance: TP/FN/TN/FP changes from26/11/30/40 to35/2/10/60. Its balanced agreement decreases about2.12 points. Keep that failure visible; lower CE and improved call metrics alone are insufficient for promotion. No new step81 game grid was registered; the next16-game evaluation is162, and the sole candidate remains final324 with64 games.

## Preserve the acquired replay mixture

The informative attack dataset retains all22,959 TRAIN and5,905 validation decisions, with2,755/879 whole-game groups. Every20,237/4,721 expert record from the calls course remains: identical actual JevAny API request, observed label, target and source group. All stage1 ancestors and reserved groups are excluded from held-out overlap. The only changes are loss-only source tags outside model input.

Use a proposed75% expert /25% conditional-simulation loss mass. Within expert replay, preserve the calling course's relative plain25%, riichi10%, open25%, call25%, kan15% mixture. Both riichi classes keep their natural194:272 TRAIN ratio and equal per-example weight. Give each of the four MC phases6.25% total mass. No example is copied, and no label or confidence is changed.

| Loss stratum | TRAIN | Validation | Total loss mass |
|---|---:|---:|---:|
| Expert ordinary discard |8,826|1,865|18.75%|
| Expert riichi accept |194|37|3.1223%|
| Expert riichi decline |272|70|4.3777%|
| Expert open discard |5,590|1,220|18.75%|
| Expert call window |3,569|961|18.75%|
| Expert kan window |1,786|568|11.25%|
| MC closed discard |717|457|6.25%|
| MC open discard |254|100|6.25%|
| MC call window |1,503|477|6.25%|
| MC kan window |248|150|6.25%|

The largest per-example coefficient is5.7861, below8. The count-only scheduler proposal is accumulation127:181 updates per complete epoch, with99 records in the last update,77.95% of a full group. Two diagnostic permutations retain every row exactly once per epoch and include all10 strata in each update. These are sampler checks, not362 training updates; the actual stage3 epoch budget remains unregistered.

The loss adapter restores each MC question to the canonical `paired_future_masked` source before computing soft-target loss. Unobserved legal actions therefore stay unlabelled and receive zero gradient. Treating the new source string as an ordinary soft label would silently lose that mask. The actual preflight compares160 real losses and gradients,16 from every stratum, against the pinned research implementation. Maximum difference is0;34 sampled menus have unlabelled legal alternatives, whose gradients remain zero. The [source/uncertainty audit](curriculum-stage3-provenance-20261121.md) still governs interpretation of these fixed16/32-world labels.

## A separate calls-final parent bridge

`tools/curriculum_calls_parent.py` leaves the frozen stage1 bridge untouched. It only accepts the registered plain-input calls final after its completed fit, full4,721-menu reload,376 optimizer-state checks,224 reconciled games including references, original stage2 quality gate, exact replay and ancestral group checks. It recomputes the quality metrics rather than trusting a passing flag. No minimum-loss or intermediate-checkpoint selection is admitted. The future child must load the complete same adapter/head with a new optimizer through the pinned initialization/lineage checks.

The seal distinguishes local step324 from cumulative step908. Successful calls fitting alone is insufficient: its original advancement targets must pass and the audit must exit successfully. A context-aware calls parent needs an additional versioned serving-contract audit; this bridge rejects it explicitly. The failed original stage1 grade remains preserved in the sealed lineage.

Slurm can expire completed job history. The new bridge permits a previously captured, actual COMPLETED/0:0 scheduler observation when the current query specifically reports an invalid job ID. A PENDING/RUNNING snapshot, a shell exit alone, a transient controller failure or a currently running job never qualifies. The chosen immutable snapshot is copied into the seal; an append-only observation journal is not frozen as a live dependency. This avoids restarting completed training merely because scheduler history disappeared.

Seven focused regression checks passed inside the approved GPU allocation. They cover mask preservation, replay mass and natural riichi ratio, unexpected loss sources, actual-quality failure despite a passing flag, partial/reload/small-grid rejection and expired scheduler-history handling. The real prepared-data preflight also passed exact API parity, pinned ancestral group checks,160 loss/gradient comparisons and two complete sampler epochs. **A positive full calls-parent seal/load and actual stage3 tensor warm start are still pending the real completed stage2 result.** These checks are not a trained attack policy or a strength result.

## Reproduction and identities

After the public informative pool and calls data are prepared, run inside an approved scheduler GPU allocation:

```bash
python tools/prepare_curriculum_attack_course.py \
  --data-ready runs/curriculum-optimization-v2/stage3-data/data-ready.json \
  --replay-ready runs/curriculum-optimization-v2/stage2-data/data-ready.json \
  --reserved-groups runs/curriculum-reproduction-v1/clean-data-v1/reserved-public-groups.json \
  --out runs/curriculum-stage3-attack-v1/data
python tools/check_curriculum_attack_course.py \
  --registered runs/curriculum-stage2-calls-v1/fit-plan/registered.json \
  --data-ready runs/curriculum-stage3-attack-v1/data/data-ready.json \
  --out runs/curriculum-stage3-attack-v1/precheck-private.json
python -m unittest tests.core.test_curriculum_attack_course tests.core.test_curriculum_calls_parent -v
```

Our preparation additionally binds `--world-audit runs/curriculum-optimization-v2/stage3-world-audit-v1/summary-private.json`. That optional private evidence does not change public data bytes or labels. Without it, provenance verification is disclosed as absent; it is never invented. The preflight requires an actual calls protocol for the lineage checks and refuses existing output/order-receipt paths. It does not use unfinished calls weights as a teacher.

| Prepared file | SHA256 |
|---|---|
| Attack TRAIN |`9eacc26d02f25c732f22fb1007c4a5601cbe36b4f12e92f8f43dc67a7c7484e5`|
| Attack validation |`892a3bd543d9824cff01b9720460abf0f2823201798b52a2f81493559e591838`|

Future registration still needs a verified calls final, a frozen attack optimizer/exposure/evaluation recipe, actual parent/tensor initialization checks and a full attack-course audit. Stage4 full-match public-information labels/training and the reserved unseen v8 comparison remain outstanding. New data records, source links, order diagnostics, per-world details and weights stay private; only code, tests and this aggregate summary are published.
