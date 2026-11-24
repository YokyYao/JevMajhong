# Placement continuation retains the qualified parent's replay objective

The previous placement wrapper delegates inherited rows to the old3B loss,
which would restore the replaced basic-RI formula in the retained-parent
lineage. A distinct prospective placement objective now retains all25
inherited objectives exactly and leaves the32 new terminal objectives intact.
No placement parent, weights, optimizer or fit is admitted by this change.

The first full-data preflight exposed a real annotation issue: upstream
normalizes soft weights, while the old terminal annotator requires bitwise
equality to their raw representation. The corrected v2 interface follows the
upstream conversion, option order, sum and division for an annotation-only
derived question. It requires exact agreement with the actual materialized
target. Raw teacher labels and encoded inputs remain unchanged. The original
failed preflight, source and failure evidence are preserved privately.

Actual allocated-node v2 evidence:

| Check | Result |
|---|---:|
| Every currently prepared TRAIN row |91647|
| Exact retained3B replay rows |91247|
| Existing terminal rows |400|
| Actual source types checked |48 of the required57|
| Maximum objective/standard-CE difference |0|
| Representative source gradients checked |48|
| Maximum representative gradient difference |0|
| Rows requiring annotation normalization alignment |2|
| Maximum raw-to-materialized target difference |1.11e-16|

The complete available-data proof took77.40 seconds with synthetic logits,
without a model, optimizer update or game. Five corrected v2 loss tests and
five v2 validation-metric tests pass. Tests preserve zero gradient for untested
non-RI alternatives, reject partial RI targets and altered materialized targets,
and retain all25 inherited gradients. The validation adapter keeps the same
terminal selectors/diagnostics and explicit current-file membership, records
ten components and signed objective-minus-CE, and still requires all57 source
pools. Low masked loss is not reported as a win/top-two probability.

The original80% replay/20% terminal source masses, count-only repeats and
scalar/repeat caps are unchanged. Uniform scalars in the software gradient
proof are not registered production weights. Nine targeted win/reach/ankan/
kakan context pools are still collecting; they are not imputed or removed.
Full57-source data/validation, mixture/exposure accounting, qualified3B parent,
trainer/auditor integration and actual placement training remain required.

No active3A source, data, optimizer or curve was changed. The unopened final
comparison, main and v8/Windows remain unchanged. Only source, tests, plans and
this aggregate are public. See the [retained-replay plan](plans/curriculum-placement-parent-loss-20261124.md)
and [exact normalization plan](plans/curriculum-placement-normalized-alignment-20261124.md).
