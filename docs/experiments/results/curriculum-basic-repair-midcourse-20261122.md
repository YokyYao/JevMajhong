# Basic repair: the first trained complete-game point and unresolved riichi bias

2026-11-22. The registered fresh-Qwen basic fit is still running, with at least
170/608 actual optimizer updates observed. This is interim development
evidence. The sole candidate remains the registered final608, not this midpoint;
the complete four-course/v8 objective is not achieved.

GPU315539 completed on ws-l1-011 in1m05s, scheduler/exit0:0. It ran164 curriculum
checks, exact replay/profile controls, a read-only matched-match audit and the
predeclared0.5 class-posterior diagnostic at152. An earlier allocation315535
exited at its GPU occupancy preflight before running these computations; the
retry used a different allocated node. Neither job changed the active fit,
its data/configuration or any169 historical recovery bindings.

## Actual matched sixteen-match development result

All rows below use exactly the same two logical seeds, four seats and two
opponent profiles. The64-match reference caches were restricted to these
sixteen matches, not compared by their full64 summary. Completion, final scores,
settlement ledgers, ranks, wins/deal-ins, grid coverage and checkpoint identities
were checked from the actual saved logs. No new model games were played.

| Policy | Net points/match | Top two | First | Fourth | Hand win | Deal-in |
|---|---:|---:|---:|---:|---:|---:|
| Fresh initial0 |−11,437.50|6.25%|0%|68.75%|6.16%|32.47%|
| Earlier completed plain584 control |−1,193.75|62.50%|25.00%|18.75%|20.07%|12.86%|
| Frozen v8 |+3,612.50|62.50%|62.50%|31.25%|22.37%|10.34%|
| Current basic repair152 |+4,756.25|50.00%|50.00%|0%|20.91%|12.87%|

Current152 gains+1,143.75 net points/match over matched v8, but loses12.5
percentage points of top-two and has2.53 points higher deal-in. Its native
match reward is0.59453125 (terminal raw net/8000), a real evaluation outcome;
supervised training reward remains null. These two seed groups are too few for
an independent strength claim. A positive net mean, or0 fourths in16 matches,
does not establish v8 parity or justify overriding the basic-source diagnostic.

## Validation loss falls while riichi refusal collapses

| Validation point | Weighted CE | Legacy accept | Legacy correct refusal | Teacher accept | Teacher correct refusal |
|---|---:|---:|---:|---:|---:|
|76|1.652667|28/37|13/70|230/287|67/238|
|152|1.554856|37/37|2/70|285/287|7/238|

Initial weighted CE was2.472033. At152 ordinary-discard exact agreement is
44.15%, with plain CE1.698577. Of632 riichi windows, the model selects reach in
621:322 true accepts and299 false accepts. Exact ordinary-discard agreement,
separate source quality and complete-game results must stay visible alongside
the lower average CE. No parent has passed the source-specific admission gate.

Using the already fixed0.5 full-menu class-posterior rule instead of individual
action argmax yields legacy TP35/FN2/TN4/FP66 and teacher TP281/FN6/TN19/FP219.
Class-probability AUC is0.57915/0.58783 for legacy/teacher. Actual native guarded
argmax has the same teacher riichi-class confusion as raw argmax. Thus this
point is not explained away by only pooling discard alternatives or by the
existing native eligibility filter. It does not identify a single causal
training defect, and no threshold was searched, selected or deployed.

The repeated full-source eligibility audit still finds317/4,993 TRAIN and
99/1,294 validation teacher targets outside the native guard's allowed set.
That mismatch remains relevant to future risk/pursuit training but does not
explain this point's unchanged guarded/raw riichi-class counts. This checks the
native evaluation helper, not Windows screenshot/serving equivalence.

## Reusable next-course controls, not a trained child

`tools/curriculum_calls_repair.py` now supplies twelve-source validation,
separate operation accept/decline counts by source, the unchanged six-source
basic-only profile and a stronger parent-loading bridge. Aggregate-only seals
and old research overrides are rejected. A completed source-qualified basic
parent must still pass the original model/protocol/replay/new-optimizer checks.
An actual qualified parent load remains pending; fixture tests do not establish
that a running parent is qualified.

The actual prepared48,710/12,404 child rows retain every14,285/3,266 basic input,
label and group. Two complete254-record sampler permutations retain all twelve
strata per update without copies. Reprojecting the actual basic validation rows
and reusing the saved152 predictions reproduces every original source CE,
agreement/count/confusion metric exactly. All3,266 metadata-only replay-control
probabilities match with error0. This is a projection/control check, not a new
child's tensor-load or inference result. Full372-adapter/four-head identity,
real child step-zero prediction parity and full child registrar/trainer/auditor
are still required before a later fit.

Continue the fixed608/four-epoch basic run and inspect the next actual228
validation/304 same16 game point. Preserve complete real loss/reward/win/top-two/
net curves and the final64 audit. Do not stop merely because152 is not a
quality pass, select its favorable net mean, silently change decoding/loss,
start calls from an unqualified midpoint or open the final confirmation grid.
Source/plans/aggregate evidence may be public; new raw decisions, teacher
labels, predictions, Q details and weights stay private.

```bash
python -m unittest discover -s tests/core -p 'test_curriculum*.py'
python tools/audit_curriculum_development_point.py \
  --registered runs/curriculum-basic-repair-v1/fit-plan/registered.json \
  --step 152 \
  --control runs/curriculum-optimization-v2/stage1-fit/evaluations/step-000584 \
  --out runs/your-fresh-interim-audit
python tools/audit_curriculum_riichi_execution.py \
  --registered runs/curriculum-basic-repair-v1/fit-plan/registered.json \
  --basic runs/curriculum-basic-teacher-data-v1/data --step 152 \
  --out runs/your-fresh-riichi-audit
```

Run only in bounded scheduler GPU allocations. The audit commands require the
corresponding private source/cache artifacts; they are not downloadable public
test data.

## Update: completed228 validation, still no new game point

The full3,266-menu validation at228 completed with weighted CE1.298851,
ordinary-discard agreement48.37% and plain CE1.403418. Legacy confusion is
TP8/FN29/TN56/FP14; teacher confusion is TP86/FN201/TN204/FP34. Aggregate
riichi acceptance is94/324=29.01%, correct refusal260/308=84.42%. Compare152's
322/324=99.38% acceptance and9/308=2.92% correct refusal: the bias switches
direction rather than demonstrating stable mastery. The source-specific grade
still fails. Exact decline-discard agreement62.01% is a different metric from
correctly choosing the non-riichi class84.42%; do not conflate them.

No228 game evaluation was registered; the next matched16 point is304. Lower
validation CE alone does not establish better match reward, top-two or net
points. The fixed0.5 class/eligibility diagnostic can be repeated on the saved
228 probabilities without new labels/inference/games or changing active action
selection. The original608 final candidate and four full epochs remain fixed.

GPU315639 subsequently completes the fixed diagnostic with exit0:0. Riichi
probability AUC improves from152's0.579/0.588 to228's0.648/0.658 for legacy/
teacher. This is evidence of improved ranking on these held-out examples,
alongside an unstable actual action threshold; it does not establish a single
cause or better match strength. Fixed full-menu0.5 class decoding gives legacy
TP3/FN34/TN65/FP5 and teacher TP35/FN252/TN228/FP10, missing still more positive
cases. Native guarded teacher argmax gives TP88/FN199/TN201/FP37, only a small
change from raw86/201/204/34. No diagnostic decoder/threshold was deployed.
