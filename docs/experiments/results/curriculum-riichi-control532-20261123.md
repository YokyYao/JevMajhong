# RI control532: refusal recovers, but qualification still fails

2026-11-23, Asia/Dubai. Actual532 validation and an independent cached audit
are complete. The same fixed608/four-epoch training continues.532 has no native
game grid: the latest real strength remains the negative456 matched16 result.
No parent, later course, production model or final-test result is admitted.

| Validation | Max-class380 | Max-class456 | Max-class532 | Hard CE532 |
|---|---:|---:|---:|---:|
| Common weighted original-action CE |1.199252|1.180501|1.170951|1.175244|
| Actual changed objective |1.195975|1.186678|1.170818|not this run's objective|
| Ordinary-discard agreement |51.56%|52.32%|52.39%|52.66%|
| RI accept recall |219/324 (67.59%)|274/324 (84.57%)|246/324 (75.93%)|280/324 (86.42%)|
| RI decline recall |184/308 (59.74%)|111/308 (36.04%)|165/308 (53.57%)|105/308 (34.09%)|

532 improves refusal from456, but all source checks still fail: legacy accepts
27/37 and correctly declines37/70 (52.86%); teacher accepts219/287 and declines
128/238 (53.78%). Both refusal recalls remain below the unchanged0.6 limit.
Aggregate=false, legacy=false, teacher=false, all-source=false. Lower loss and
ordinary agreement do not establish RI stability or playing strength.

## The same cases reveal a one-direction change

| Same632 RI cases,456 to532 | Legacy | Teacher | Pooled |
|---|---:|---:|---:|
| Correct remains correct |49|308|357|
| Correct becomes wrong |8|20|28|
| Wrong becomes correct |15|39|54|
| Wrong remains wrong |35|158|193|
| Changed RI decisions |23|59|82|
| Net correct change |+7|+19|+26|

Correct total recovers385→411, still below304's413. All82 class flips are
reach→discard:54 formerly incorrect reaches become correct refusals, while28
formerly correct reaches become misses. There are no discard→reach flips in
this interval. This follows from the fixed binary labels, before/after source
confusions and the transition counts; it is not a claim about exact tile-choice
changes within the discard class. The result is consistent with a reduction
in reach propensity on these cases, not proof of its cause or stable learning
of more detailed conditions.

Teacher argmax-margin AUC falls slightly from456's0.7575 to0.7507, still above
same-step Hard CE0.7226. Mean saved reach probability falls from0.6002 to0.5524
on accepted labels and0.4557 to0.3976 on refused labels. These are correlated
development diagnostics, not calibrated winning probabilities or independent
evidence that a decoder or training change will improve play. Actual RI-positive
loss mass over457–532 is0.498790; this is loss weight, not measured gradients.
No threshold is tuned, source gate waived or favorable midpoint promoted.

## Scope of the completed audit

GPU316254 on ws-l1-005 completes0:0 in14s. Its pending begin time is advanced
only after the complete532 validation metric exists; code, source, resources
and main training numerics are unchanged. Actual scheduler/shell exits are
archived. The previously tested same-case helper is reused without a new
unit-test-run claim, model inference, new labels, games or optimizer updates.

Both eight-point cached3,266-menu validation prefixes, raw source confusions,
checkpoint identities and the earlier independent304 anchor reconcile. Every
actual1–532 source count and objective/CE agrees with the frozen orders and
same-update original control. Four generated epoch orders are checked, but
only three full epochs plus76 updates of the fourth are consumed here:
49,999 actual exposures. Maximum objective/common-CE reconstruction errors are
7.07e-8/4.44e-16. Do not describe this as four finished epochs.

Real curves are generated privately with532 train points, validation
0/76/152/228/304/380/456/532 and game points0/152/304/456. No532 game is inferred
or interpolated; final64 is still absent. The complete final export will retain
this532 point. This interim report publishes only aggregates rather than another
copy of the full curve files. Existing [456 curves and actual games](curriculum-riichi-control456-20261123.md)
remain unchanged: max-class−6,643.75 net/match and12.50% top two, versus Hard
CE456−5,393.75/31.25% and the matching v8 subset+3,612.50/62.50%. Only two
consumed seed clusters are present; these are not independent final results.

Class532 identitya27980b871ba2e4ba640fd9f5c2c3bb1570723230bfee654527babebf3f61494;
Hard CE532a5c777e726543cc77661b75f820858200d51da902936ccfbf235b91b8924bbd1.
All successful prior bindings remain unchanged. Main315768 continues to608;
315769 full independent final audit and315770 terminal capture remain dependent.
The [final-only fixed decoder contrast](../plans/curriculum-riichi-final-decoder-contrast-20261123.md)
is tested and queued as316338 after315770, but has not run. Its hypothetical
static agreement will not override actual policy outcomes or admit a parent.

The full four-course objective remains unachieved. Later calls/risk/placement
fits and the unopened32-seed/256-match-per-policy v8 comparison are required.
All computation uses bounded scheduler GPU allocations with companion CPUs,
resource checks, logs/timeouts/actual exits. Raw hands, labels, case transitions,
scores, event journals and new weights remain private. No main merge or
production/Windows replacement follows this interim recovery.
