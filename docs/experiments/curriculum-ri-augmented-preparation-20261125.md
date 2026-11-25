# Audited augmented RI corpus and retention preparation

This is a separately frozen standby data/optimizer preparation. It does not change
the continuing2472-update trial or qualify a replacement for v8. That trial's
declared full result and independent audit still decide whether it advances3B
or may trigger a later alternative.

## Complete data and source mixture

| Item | Continuing trial | Prepared standby corpus |
|---|---:|---:|
|Unique TRAIN decisions|87595|92570|
|VAL decisions|23688|24998|
|Basic-replay RI TRAIN opportunities|2353|4244|
|Physical record exposures per epoch|90228|94754|
|Optimizer updates per epoch|1236|1298|
|Two-epoch update budget|2472|2596(prospective)|

The original source JSONL line bytes remain an exact prefix of each union.
All4975 new TRAIN and1310 new VAL rows come from the separately completed and
audited256-family Mortal cohort. Their normalized public states, labels, masks,
identities and whole-family splits are preserved; only the private loss-source
tag changes. Full union IDs, family groups and visible inputs are disjoint across
TRAIN/VAL. Existing original same-split rows are retained rather than deduplicated
or relabeled. Original23688-menu VAL and new1310-menu VAL remain explicit separate
cohorts for future retention reporting.

Keep all19 source coefficients. Three teacher basic-source counts change:

| TRAIN stratum | Original | Augmented |
|---|---:|---:|
|Teacher plain discard|3106|6190|
|Teacher RI accept|1075|2123|
|Teacher RI decline|812|1655|
|Legacy RI accept|194|194|
|Legacy RI decline|272|272|

The new teacher data adds independent situations without increasing its total
origin/class mixture coefficient. The prospective KL term is a separate objective
change, so this is not an isolated data-only experiment or a claim of equal
optimization trajectories.

## Full coverage without dropping excess teacher RI

The old exactly-one-per-RI-source schedule rejects an incomplete plan if a source
contains more originals than there are updates. It remains valid for the current
smaller corpus. The new schedule admits at least one member of every RI source
per update and evenly inserts additional teacher originals. All92570 originals
appear in each epoch. Legacy RI occurs once per update; each teacher RI source
occurs once or twice. Every update has73 physical records.

Each epoch uses94754 physical exposures, including2184 repeated slots; only54
padding slots repeat non-RI examples. The actual maximum repeat is7 under cap8;
the maximum per-example weight is9.551815 under cap16. Inverse-repeat weights
preserve each original's exact within-source empirical objective contribution;
maximum numerical mass error is below1.4e-20. The existing generic epoch adapter
accepts both complete frozen plans. Four relevant schedule tests passed, including
the excess-RI case that the old sampler correctly rejects.

## TRAIN-only retention probabilities

All4244 parent probability menus are complete and frozen. Reuse2353 original
menus only after their source/model bindings pass; infer1891 new TRAIN RI menus
with the exact same qualified calls-course initializer. Complete input/label/source
and candidate-menu membership matches the augmented TRAIN set. No VAL/test request
is inferred to create targets; no optimizer exists during cache generation, and
the teacher bundle stays unchanged.

Mortal supplies the fresh hard action labels. The qualified earlier student
supplies the separate soft probabilities that constrain RI drift. The latter is
not an optimal teacher and can preserve mistakes; its usefulness still requires
actual training and native outcomes. Four additional weighted-objective/observer
tests passed, checking independent KL gradients, exact unchanged non-RI weighted
gradients, reconstruction of the optimized loss and malformed-input rejection.

The first full-model integration attempt admitted all92570 unique inputs but
failed before its first optimizer update:the pinned augmentation removes `_meta`,
where the initial materialization hook expected to find the original sample ID.
Its original source, log, registration and successful admission receipt are
preserved privately. This was a new integration error, not evidence about the
ongoing trial's strategy quality. No control checkpoint was produced.

The repaired epoch adapter attaches the target only after the actual encoder
returns, using the existing validated virtual-position -> original-row mapping.
It requires unchanged source/label/menu and inverse-repeat identity, and leaves
the pinned augmentation and token encoding intact. Three regression tests cover
metadata stripping, all physical epoch positions and rejection of missing or
repeated targets. The composed mathematical loss itself is unchanged.

The isolated repaired full-model control is checking combined input admission,
two real optimizer updates, complete optimizer state and saved adapter/head
prediction reload. It cannot qualify a strategy and must never initialize a
future course. Full fit registration, original source-quality/native
criteria, actual milestone loss/reward/win/top-two/net curves and the unopened v8
confirmation remain unfinished. See the [fixed data recipe](plans/curriculum-ri-augmented-data-20261125.md)
and [prospective retention objective](plans/curriculum-ri-parent-anchor-20261125.md).
Only reusable code, recipes and aggregate observations are public.
