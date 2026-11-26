# Stage4 TRAIN inventory and prospective budget

The complete targeted **TRAIN** population can already be accounted for using
ended writers. This is a count-only inventory and budget proposal. It does not
admit a student fit, claim the full targeted validation collection is complete,
or replace the original full-data registration.

The [57-source CSV](curriculum-placement-train-inventory-20261126.csv) records
capability groups, actual unique TRAIN records and source families, existing
validation counts, planned targeted validation counts, source loss masses,
virtual repetitions, and scalar loss weights. Counts of families in separate
sources overlap and must not be summed as globally independent families.

| Population | Records |
| --- | ---: |
| Previously completed coherent placement TRAIN base | 91,647 |
| Complete new targeted TRAIN labels from ended writers | 71 |
| Total unique TRAIN records | **91,718** |
| Explicit sampling quarantine among the 72 fixed targeted TRAIN views | 1 |
| Existing validation records | 24,607 |
| Fixed targeted validation views still requiring full collection/admission | 36 |

If all 36 targeted validation views produce complete labels, the resulting
validation corpus will have 24,643 records. Explicit sampler quarantine may
reduce that total; the actual complete merge determines the final count. The
planned validation column in the CSV is not a claim that all these labels have
already been admitted.

The first two completed targeted shards account for 53 TRAIN labels and one
quarantine. The third shard's original ended writer saved 21 complete cases,
including its 18 TRAIN views. Its actual recovery admission and original failed
terminal establish those already completed cases; the failed whole job is not
represented as successful. No currently appended label file was used for this
inventory, and no raw case worlds were replayed or rescanned.

## Actual source mass, distinct records and repeated exposures

`retained25` means **25 retained source categories**, not 25% replay. The existing
57-source placement recipe assigns **80% of the normalized source loss mass to
those 25 retained categories and 20% to 32 terminal-placement categories**. Each
terminal category receives 0.625%. These are loss masses, not raw row ratios or
claims that every optimizer update includes all categories.

The original retained objectives remain in use: ordinary/riichi, calls,
post-call/riichi continuation, counterfactual efficiency/defense, and the six yaku
or control categories. Terminal categories include win accept/decline contexts,
non-win decisions, win controls, and complete win/riichi/ankan/kakan menus.

The unchanged bounded expansion formula yields **92,612 record exposures per
pass**, including **894 explicit extra exposures**. Repetitions preserve the
original input, target and family and add no independent samples. No view was
filtered according to action advantage, label confidence or positive reward.

## Prospective schedule from the actual TRAIN counts

| Parameter | Count-derived proposal |
| --- | ---: |
| Epochs | 2 |
| Batch size | 1 |
| Gradient accumulation | 75 |
| Updates per epoch | 1,235 |
| Total updates | **2,470** |
| Total record exposures | 185,224 |
| Final update's records per epoch | 62 |
| Curve/evaluation milestones | 0, 700, 1,400, 2,100, 2,470 |

These values come from the existing `layout` and `proposed_schedule` functions,
not a newly chosen shorter training budget. The actual complete-data preparation
and registration must independently reproduce and freeze the budget before any
optimizer update. The existing learning rates, fresh optimizer, completed 3B
parent, full validation requirement and final candidate criteria are unchanged.
The independent final frozen-v8 confirmation has not been opened.

## Validation of this inventory

The final capability classification and aggregate export completed in a finite
GPU job with actual scheduler state `COMPLETED`, exit `0:0`. The program verifies
the completed base counts and bound original source selection, exact ended
writer file identities, all 72 targeted TRAIN source IDs including quarantine,
source/family/split membership, all 57 categories, and the unchanged count-only
mass/repeat/budget arithmetic. It performs no model loading, simulation or
optimizer updates. This is not evidence of a trained Stage4 checkpoint.

Implementation: `tools/summarize_curriculum_placement_train_inventory_v3.py`.
Only the aggregate CSV, this note and reusable code are public; original source
rows, worlds, case records, model weights and private execution receipts remain
private.
