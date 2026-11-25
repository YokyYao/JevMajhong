# TRAIN-only call coverage behind the remaining acceptance gap

After the completed 3A2400 result, a finite companion step in GPU allocation
324570 counted all 87,595 registered TRAIN records without loading a model,
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

Before choosing a call-specific revision, measure the existing TRAIN positive
examples with the parent and final model to distinguish failure to fit known
positives from failure to generalize. That model-inference diagnostic has not
been run by this accounting step. The separately queued expanded real-teacher
kan-window diagnostic must likewise report actual outcomes; its future output
cannot be assumed to contain positives.

Prioritize independently generated, family-separated situations with a credible
reason to call **and** counterexamples where passing is better. Keep provenance,
operation, hard/soft source and parent scenario families as reusable data
categories. Evaluate acceptance and decline by operation/source, then complete
matched games; neither higher call frequency nor lower label loss alone proves
better Mahjong. The already used validation grid remains development data and
the reserved final grid stays unopened.

The accounting source and private binding are retained locally. This public
summary exposes aggregate counts only, without raw hands, labels or model traces.
