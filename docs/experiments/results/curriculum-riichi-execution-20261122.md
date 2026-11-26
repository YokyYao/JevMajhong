# 76-step riichi diagnosis: class ambiguity and execution constraints

2026-11-22. The registered fresh basic fit continues toward608; its configuration, weights-in-use, inference route and original quality controls are unchanged. This is a consumed-development diagnostic, not a candidate selection, win probability, final game result or v8-strength claim.

At76 steps, actual validation loss falls from2.472033 to1.652667. Plain action agreement is31.97%. Riichi accept/decline recalls are28/37 and13/70 on unchanged legacy validation, and230/287 and67/238 on teacher validation. Thus lower CE accompanies persistent over-selection of reach; the primary problem is not yet solved.

## Fixed cached-probability diagnostic

GPU315495 completes with exit0:0, using only cached76 predictions and exact native reconstructions of the6,287 new basic source rows. Its [diagnostic formulas](../plans/curriculum-riichi-execution-audit-20261122.md) are fixed before calculation; no threshold sweep, new model inference, optimizer update or policy game is run. All active fit inputs and the76 checkpoint remain unchanged.

| Source / diagnostic | TP | FN | TN | FP |
|---|---:|---:|---:|---:|
| Legacy raw argmax |28|9|13|57|
| Legacy fixed P(reach)≥0.5 |23|14|43|27|
| Teacher raw argmax |230|57|67|171|
| Teacher fixed P(reach)≥0.5 |152|135|143|95|
| Teacher actual guarded argmax |230|57|67|171|
| Teacher fixed eligible-renormalized P(reach)≥0.5 |162|125|112|126|

Reach is one candidate while non-reach can distribute probability across several discards. Summing class probability changes the observed bias, but trades acceptance for rejection rather than proving a generally better policy. Reach-probability rank AUC is only0.61390 on legacy and0.58266 on teacher windows. Therefore simple threshold/routing changes cannot be presented as a demonstrated solution to weak discrimination. No diagnostic rule is selected for deployment or substituted for the original raw-argmax gate. Teacher guarded/raw riichi class counts coincide at this checkpoint, so the guard does not explain these particular class errors.

## Actual serving-helper target exclusions

The pinned native evaluation PolicyAgent scores the full legal menu, then filters choices through `guarded_indices` plus empty-feature special actions. A legal teacher target can be excluded by that subsequent filter. Actual native replay finds:

| New source split | All targets excluded | Ordinary targets excluded | Riichi-declined targets excluded | Riichi-selected targets excluded |
|---|---:|---:|---:|---:|
| TRAIN |317/4,993|240/3,106|77/812|0/1,075|
| Validation |99/1,294|77/769|22/238|0/287|

Most exclusions face opponent riichi, though45 TRAIN/12 validation exclusions occur with no opponent riichi. These counts prove an input-label/execution-eligibility mismatch on the native evaluation route; they do not prove every excluded teacher move is optimal, nor certify Windows screenshot/click behavior. The mismatch can limit transfer from improved imitation to actual choices and is also relevant to later risk/pursuit data whose first simulated action can be forced outside the normal serving filter.

Keep the current full fit and real curves intact. At later fixed checkpoints, distinguish weak score discrimination, action/class probability concentration and actual execution exclusions. Any proposed loss, input or serving-contract change needs its own version and controlled full-game comparison; removing a guard or using0.5 from this table is not justified as an automatic live fix. Future terminal labels must match the candidate's actual execution action set, while the full objective must still allow learned risk/pursuit rather than silently narrowing it to an older guard's choices.

Reproduce inside a bounded GPU allocation after76/cache/source preparation, with a fresh output:

```bash
python tools/audit_curriculum_riichi_execution.py \
  --registered runs/curriculum-basic-repair-v1/fit-plan/registered.json \
  --basic runs/curriculum-basic-teacher-data-v1/data \
  --out runs/curriculum-riichi-execution-audit-v1/audit-v1 --step 76
```

Only code, this aggregate report and the plan are public. Cached per-record predictions, excluded target lists, raw observations and new weights remain private. Main, practical v8 and Windows remain unchanged; the final reserved-v8 comparison is unopened.
