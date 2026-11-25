# TRAIN-only call fit: existing positives are still poorly matched

A fixed training-set diagnostic compared the qualified calls388 parent and
the completed retained3A2400 model. For each of five operations, it selected
64 hard-label accept and 64 hard-label decline examples before predictions,
round-robin across available source strata. Overlapping windows were inferred
once, giving **590 distinct TRAIN rows and 640 operation/class memberships**.
Selection used a fixed ID hash seed, not model errors. Both frozen models used
the same pinned serving runtime and complete-legal RI0.5 selection policy.

| Operation | Parent correct accept /64 | Final correct accept /64 | Parent correct decline /64 | Final correct decline /64 |
|---|---:|---:|---:|---:|
|Chi|18|18|64|64|
|Pon|18|21|59|58|
|Ankan|35|40|47|40|
|Daiminkan|0|5|64|57|
|Kakan|30|36|39|38|

The severe daiminkan gap is visible even among previously trained positives:
the final model matches only **5/64 (7.81%)** sampled positive labels, versus
0/64 for its parent. Chi remains 18/64. Thus the held-out shortfall cannot be
attributed solely to encountering unseen situations. The existing positive
supervision is poorly matched too. This does not establish why it is poorly
matched: insufficient fitting, source conflicts and imperfect labels remain
possible. The legacy positives are observed actions with traced provenance,
not certified optimal actions; see the
[earlier full operation-label audit](curriculum-operation-label-balance-20261124.md).

This deliberately balanced/source-stratified TRAIN panel is **not** held-out
accuracy, a natural call frequency or a strength evaluation. No validation/test
rows were read, no new games were simulated, and no model was updated. Parameter
bundle identities were checked before and after inference. The finite companion
GPU step completed with exit0; the ongoing 2,472-update trial continued separately.

Saved-menu re-accounting found no win candidates in the 590 rows, so a missing
forced-win shortcut does not explain this panel. On its 64 daiminkan-positive
labels, the final model chose pass59 times and daiminkan5 times; its parent
passed64 times. Mean daiminkan preference increased from0.237 to0.301 on
positives, but also from0.212 to0.264 on negatives, where the final model made
seven erroneous calls. These are model action preferences, not winning
probabilities. Increasing willingness to call without improving discrimination
can hurt the decline cases too; this is not a justification for lowering an
execution threshold.

## Re-accounted training coverage

After the completed 3A2400 result, a finite companion step in an allocated
GPU job counted all 87,595 registered TRAIN records without loading a model,
reading validation/test data or changing the active trial. These are source-row
counts and canonical loss-mass summaries, not independent simulated situations,
new teacher labels or evidence of strategic optimality.

An opportunity means the actual candidate menu contains that operation and at
least one alternative. Hard labels are split into operation acceptance and
another action; soft-target rows remain separate. A row with multiple operation
types can appear in multiple operation totals.

| Operation | Opportunity rows | Hard accept | Hard decline | Soft-target rows | Accept fraction of canonical window target mass |
|---|---:|---:|---:|---:|---:|
|Chi|12,026|1,444|9,709|873|25.81%|
|Pon|8,193|2,040|5,142|1,011|25.99%|
|Ankan|1,292|501|693|98|32.86%|
|Daiminkan|1,052|341|602|109|24.98%|
|Kakan|421|157|223|41|36.42%|

For the last column, each original row has canonical mass
`source_mass / unique_source_rows`. An acceptance hard label contributes one;
a soft target contributes the sum of target probability over candidates of the
operation type. Divide acceptance mass by all opportunity mass for that
operation. This describes supervision, not probability of winning or the true
natural frequency with which a strong player should call. The active weighted
replay preserves these per-original masses.

## Source-specific gap

The total data does contain positive daiminkan examples. It is therefore
incorrect to explain the 3/88 final held-out acceptance solely as "there were
no daiminkan positives in training." However, all hard positives come from the
legacy source in the current dataset:

| Daiminkan source | Hard accept | Hard decline | Soft rows |
|---|---:|---:|---:|
|Legacy kan|341|321|0|
|Teacher kan|0|281|0|
|MC kan-window|0|0|109|

The teacher kan source contributes about 1.21% of total canonical objective mass
through these daiminkan windows, entirely toward declining. Legacy positive
daiminkan labels contribute about 0.60%; MC positive target mass contributes
about 0.27%. These are category masses, not an analysis of the actual gradient
directions or evidence that the teacher is wrong. MC targets are simulations,
not expert ground truth.

Teacher chi opportunities also lean strongly toward declining: 729 accept
versus 8,142 decline in its main call source. The legacy main call source has
710 accept versus 1,522 decline. Simply adding more samples with the current
teacher-window distribution will not guarantee more varied positive calling
examples.

## Implications for the next data revision

Keep the running RI-balanced trial frozen: it tests presentation/retention and
does not claim to solve this call-coverage gap. Do not force every legal call,
relabel passes as calls, or add a kan bonus based on these counts.

The actual diagnostic above now distinguishes a problem on already seen labels
from a pure unseen-generalization explanation. It does not justify repeating
all data for more epochs indiscriminately. A call-specific revision should
separately examine operation/source/class presentation and loss mass, and the
quality of the positive/negative context pairs. The separately queued expanded
real-teacher kan-window diagnostic must likewise report actual outcomes; its
future output cannot be assumed to contain positives.

Prioritize independently generated, family-separated situations with a credible
reason to call **and** counterexamples where passing is better. Keep provenance,
operation, hard/soft source and parent scenario families as reusable data
categories. Evaluate acceptance and decline by operation/source, then complete
matched games; neither higher call frequency nor lower label loss alone proves
better Mahjong. The already used validation grid remains development data and
the reserved final grid stays unopened.

Accounting and probe sources, frozen selection, prediction traces and private
bindings are retained locally. This public summary exposes aggregate counts
only, without raw hands, labels, selected identifiers or model traces.
