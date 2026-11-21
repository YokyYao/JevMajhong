# Daiminkan data coverage, frozen-v8 diagnostic and real token admission

2026-11-21, Asia/Dubai. GPU diagnostic312242 completed on ws-l2-010 with GPU1/CPU2/48GB,20-minute limit and successful1m57s execution. Source, data, targets, active calls training and practical Windows models remain unchanged. The purpose is to investigate the calls162 daiminkan bias before committing to more data or a new attack fit.

## Counterexamples exist

| Expert replay | Daiminkan selected | Other action selected | Selected / other source groups |
|---|---:|---:|---:|
| TRAIN |341|321|325 /306|
| Validation |88|108|85 /103|

These are nearly balanced TRAIN observations. Other action can mean pon or another legal choice as well as pass. They are observed expert labels, not guarantees of optimality. This evidence does not rule out needing broader contexts, but a complete absence of negative examples cannot explain the current model behavior.

The informative MC TRAIN pool has109 daiminkan windows from51 source games, with labelled support on both daiminkan and other actions. None of these109 contrasts relies on treating an unobserved alternative as a negative. In14 windows the total target mass on daiminkan exceeds0.5; in95 it is below0.5. The target's highest individual action is daiminkan in31 cases and another action in78. These counts differ because other mass can be distributed over multiple legal alternatives. Median daiminkan mass is0.17778; median conditional bootstrap best-action mass is0.72656.

The targets remain soft and unchanged. Multiple windows can come from the same game. Bootstrap256 reuses16/32 stored futures; fixed opponent concealed hands and old continuation policies remain limitations. A selected-action nominal interval relative to the original baseline is not a calibrated confidence interval for “should kan versus should decline.” Do not count the baseline's zero self-difference as new positive evidence or turn these distributions into expert truth.

## Frozen v8 on identical consumed menus

The reference diagnostic uses1,636 expert validation menus offering chi, pon, any kan or riichi. It reloads the identity-pinned v8 and makes no new games or optimizer updates. These data may overlap v8's historical training; the result is a flat-pointer behavior diagnostic, not held-out generalization or independent strength.

| Daiminkan diagnostic | Student step162 | Frozen v8 |
|---|---:|---:|
| TP /FN |88 /0|78 /10|
| TN /FP |3 /105|67 /41|
| Observed accept recall |100.00%|88.64%|
| Observed decline recall |2.78%|62.04%|
| Balanced operation agreement |51.39%|75.34%|

The same input representation can support substantially better discrimination, as shown by the reference. This does not isolate a single training cause, prove expert decisions optimal, or establish what the student will do at final324. The [calls162 game result](curriculum-stage2-calls-20261121.md) already combines a net-point gain with greater dealin, so both imitation and complete-game risk must stay visible.

The pinned chi/pon feature helper explicitly excludes kan-containing menus. A real-data check finds771 TRAIN and247 validation daiminkan candidates, none with a chi/pon `call_features` block. A misleading closed-hand flag inherited from that helper is therefore not supported by these current records. No input-schema correction is made from that hypothesis.

## Actual tokenizer and admission

The diagnostic invokes the pinned `training_requests` and the actual `DecisionModel.encode` text path using the real fixed Qwen tokenizer. An encoding-only proxy supplies the pointer/option-isolation settings without constructing a language model. The held split is checked separately for schema/context only; no held record is mixed into an optimizer.

| Split | Requested / admitted | Rejected | Maximum state tokens | Maximum packed tokens |
|---|---:|---:|---:|---:|
| TRAIN |22,959 /22,959|0|637|1,602|
| Validation |5,905 /5,905|0|618|1,399|

Limits are4,096 state,8,192 branch and8,192 packed tokens. Record IDs/order and targets survive materialization/admission; no state is truncated. This resolves an actual entry-path question that the earlier API/CE arithmetic probes alone could not answer. It still does not execute a qualified student warm start, real training callbacks or the attack optimizer.

## Training and evaluation decision

Complete the frozen calls324/64/audit. Preserve the proposed attack75% expert replay /25% conditional-MC mixture: expert positive/negative cases and the existing MC alternatives already provide a useful first test of the risk objective. There is no evidence here to justify bulk duplication, forced kan/pass labels or a mid-fit change. Targeted fresh public-information worlds remain a later option if actual outcomes and source coverage require them.

The attack final auditor now emits acceptance, decline and balanced metrics separately for every operation, at both step0 and final. It can bind and recheck this cached v8 diagnostic through `register_curriculum_attack.py --operation-diagnostic`. The cache is checked against exact menu IDs/groups/probabilities, the current dataset/parent registration and frozen model identity; final counts are recomputed from predictions. No repeated v8 inference is needed. Without an explicitly supplied cache, the missing reference is reported rather than invented. Aggregate five-call agreement remains insufficient to claim daiminkan is solved.

This change adds diagnostic visibility and evidence binding; training data, loss masses, learning rates,362-update budget and final-only candidate rule are unchanged. The full four-course goal and reserved unseen v8 comparison remain active. No attack fit has started.

## Reproduce inside approved GPU resources

```bash
python tools/audit_curriculum_calls_data.py \
  --registered runs/curriculum-stage2-calls-v1/fit-plan/registered.json \
  --attack-ready runs/curriculum-stage3-attack-v1/data/data-ready.json \
  --out runs/curriculum-stage3-attack-v1/calls-data-audit-v1
```

Our run additionally supplied `--confidence-summary runs/curriculum-optimization-v2/stage3-world-audit-v1/summary-private.json` to inspect recovered TRAIN-only conditional uncertainty. Without those private artifacts, the public operation-contrast counts, token admission and frozen-v8 diagnostic still run; missing uncertainty is disclosed. Output paths refuse overwrite. Raw predictions, source links, confidence details and training records remain private.

The completed source run passed three operation-mask checks. Subsequent70 curriculum regressions include cached-reference identity/probability/count verification and explicit asymmetric acceptance/decline reporting. The real1,636-menu cache is also recomputed through the new auditor path without further inference. Original diagnostic logs and its successful exit are preserved. Slurm history later expired during follow-up metadata archival; that follow-up failed its strict snapshot check, and the earlier actual COMPLETED/0:0 tool receipt was preserved explicitly. Neither the diagnostic nor the main fit was restarted, and the parent-seal rules were not weakened.
