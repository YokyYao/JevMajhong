# Inverse-repeat replay: loss and journal integration

The prospective every-update RI schedule now has a separate loss interface
that applies each original decision's inverse-repeat weight and records the
actual weighted objective. It preserves all 19 existing unweighted objectives,
including inherited RI max-class/conditional-discard loss and masked MC support.
It does not reuse the old source-constant example weights for repeated rows.

The live 2,400-update trial is unchanged. No new student fit or parent selection
has occurred; this prepares a possible continuation after current-trial results
and the separately queued actual model task-gradient diagnosis.

| Completed check in scheduled GPU allocation | Result |
|---|---:|
| Original TRAIN rows materialized with retained annotations |87,595|
| Actual candidate exposure losses checked across two epochs |180,456|
| Full 73-example optimizer observations reconstructed |2,472|
| Source categories covered |19|
| Distinct source/weight logit-gradient comparisons |33|
| Materialized token encodings unchanged across those comparisons |33/33|
| Maximum weighted objective error |0|
| Maximum weighted journal loss/CE error |0|
| Unit tests |4 passed|

These checks use the real frozen TRAIN decisions, candidate schedules and
per-original weights, with synthetic logits. They are software/data integration
evidence, not observed training loss, actual model-parameter gradients, native
reward curves or stronger play. No model was loaded, no optimizer update was
performed and no simulation was generated.

Weights and original IDs are attached only to already materialized question
metadata. They are excluded from raw requests and model text. The pinned
JevAny tokenizer/encoder produced identical token encodings for representatives
of every source and all distinct source-weight combinations in both epochs.
Actual weighted logit gradients match the unchanged objective multiplied by
the intended per-original scalar.

The observer records each actual example weight. The journal aggregates
`weight * objective` and `weight * canonical CE` over consumed examples, and
retains the signed objective-minus-CE adjustment, source counts, effective
source masses and per-source weight ranges. It does not infer optimized loss
from an unweighted source mean multiplied by one constant source weight.
The full two-epoch pass checked every candidate exposure and reconstructed
all 2,472 observations without discrepancies.

Four tests additionally verify the whole empirical loss and gradient under
unequal repetitions, actual per-example weighted logging, unchanged raw
requests/unweighted losses and rejection of invalid or missing weights.

Run these commands only inside an approved finite GPU allocation after
regenerating the original corpus and candidate exposure plans:

```bash
python -m unittest tests.core.test_curriculum_ri_inverse_repeat_loss_v1
python tools/curriculum_ri_inverse_repeat_loss_v1.py
```

Next integrate the actual trainer's epoch-to-original-row mapping, record-count
and token-admission accounting, then freeze the chosen continuation checkpoint,
optimizer schedule, evaluation points and complete fit/audit entry. The source
interface alone does not start or admit that training. Preserve existing
validation and original quality criteria; the independent final comparison
remains unopened. All raw decisions, indices, weights and diagnostics remain
private; this publication contains source, tests and aggregate findings only.
