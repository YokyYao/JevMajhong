# Prospective call-type balancing within the existing thirteen sources

## Evidence and purpose

The completed3A model matches only5/64 sampled TRAIN daiminkan positives and
18/64 sampled chi positives. The original teacher kan source has no daiminkan
positives, and the expanded cohort adds395 TRAIN passes with no positives.
These findings support testing operation/class coverage and fitting, rather
than assuming another full pass through the same distribution will solve it.
Observed source actions remain imperfect supervision; increased call frequency
alone is not evidence of better play.

This preparation categorizes the **49,022 original Stage2 TRAIN rows** and
writes a reusable private per-row index/weight table. It preserves every raw
request, label, ID and family partition. No validation/test data is read and
no new labels, model updates or games are generated. The active RI-balanced
trial remains frozen and continues its registered recipe.

## Proposed numerical change

Keep all13 source loss masses exactly. Change the distribution **within four
advanced sources** only:

- Call sources: observed label classes chi, pon and other action.
- Kan sources: observed label classes ankan, daiminkan, kakan and other action.
- Every other source, including basic discard/RI, post-meld and post-RI play,
  retains its previous uniform per-original source weight.

Allocate each source's mass equally among its actually observed primary classes
and other-action class, capped so no per-row weight exceeds16. Fix capped classes
at their feasible maximum, then redistribute the remainder equally among the
remaining classes. Missing classes stay missing; no teacher daiminkan positive
is invented. Rare chi/pon labels inside a kan source remain in its other-action
class instead of being magnified as a separate tiny group.

For class count `n_sc`, total TRAIN size `N` and allocated class mass `m_sc`,
the proposed replacement scalar is `N * m_sc / n_sc`. The class masses within
each source sum to its original registered source mass. These scalars **replace**
the previous uniform source scalar; multiplying both would apply source weights
twice. The existing numerical action/RI objectives themselves are unchanged by
this proposal, but the per-original empirical objective within the four sources
is intentionally different. This is not inverse-repeat correction.

## Actual preparation results

| Source | Existing observed class counts | Proposed per-row scalars |
|---|---|---|
|Legacy call|chi710 /pon706 /other2153|1.3157 /1.3231 /0.4339|
|Teacher call|chi729 /pon1287 /other11383|1.2814 /0.7258 /0.0821|
|Legacy kan|ankan445 /daiminkan341 /kakan110 /other890|1.5744 /2.0546 /6.3691 /0.7872|
|Teacher kan|ankan56 /kakan47 /other626; daiminkan absent|16.0000 /16.0000 /1.8441|

The final teacher-kan classes are cap-constrained and therefore do not receive
equal mass. This is explicit; claiming perfect class balance would be incorrect.
All49,022 row weights were checked against all13 exact source-mass totals,
with zero copies or input edits and maximum weight16. Three tests passed:
mass conservation/missing classes, cap redistribution/infeasibility and unchanged
raw rows/unaffected sources/duplicate and soft-label rejection.

## Admission before any new training

### Completed numerical adapter checks

`tools/curriculum_call_type_loss_v1.py` now implements the replacement scalar
with all original calls13 objective components. Its batch adapter invokes the
original encoder first, then attaches loss-only metadata to the returned
materialized question; the encoded token object and raw request stay unchanged.
The scalar replaces the former uniform source scalar exactly once. Observer
accounting records source/class counts, effective mass and weighted objective
components separately from standard target CE.

The actual full-data check covered all49,022 TRAIN materializations and weighted
objectives with maximum difference0 from the intended formula. All29,539 rows
outside the four changed sources exactly preserve their old weighted objectives.
For22 observed source/class representatives, float64 **synthetic-logit gradients**
match exactly and the actual pinned tokenizer emits identical encodings before
and after annotation. Three additional tests cover loss/gradient replacement,
post-encoding attachment and corruption/double-annotation/augmentation rejection.

These checks load a tokenizer without loading backbone or student weights. They run
no optimizer update or new game and do not read validation/test data. Real full
model token admission, a saved optimizer integration control, a registered
parent/budget and eventual full policy evaluation remain pending. Passing these
checks is not evidence that the revised weighting improves Mahjong.

This artifact is a prospective data/loss recipe, not a training result. A future
adapter must verify real encoded-target alignment, replacement-scalar loss and
gradient accounting, actual source/class exposure and basic/RI preservation.
It must retain the existing operation-specific accept/decline diagnostics and
complete matched-game evaluation. A higher TRAIN match rate alone cannot qualify
a model, and existing validation observations remain development evidence.

Parent identity, optimizer budget and final candidate must be explicitly
registered before a fit. Rebuilding Stage2 would continue qualified basic608
within the fixed-base reproduction route; this preparation itself selects no
new parent and supplies no weights to the ongoing3A trial. A separate new
registration/output is required. Do not overwrite prior runs or silently add
the two new validation daiminkan positives to TRAIN.

Run preparation only in a finite scheduler GPU allocation:

```bash
python -m tools.curriculum_call_type_balance_v1
```

The index and per-example scalars stay private. Public deliverables are code,
this plan and aggregate coverage/fit reports:
[TRAIN fitting](../curriculum-train-call-coverage-20261125.md),
[expanded teacher result](../curriculum-daiminkan-context-teacher-20261125.md).
