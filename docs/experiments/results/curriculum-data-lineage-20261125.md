# Curriculum data lineage and preparation snapshot — 2026-11-25

This report records completed **data preparation and source checks** for the
reproduction course. It is not a claim that Stage 4 has trained, that the final
candidate is qualified, or that the new model matches frozen v8 in strength.

## Course and data boundaries

The selected parameter path is basic608 → calls388 → selected3A1800 → the
registered3B2500 course → the full placement course. The base dataset below is
prepared for the last course; it is not another trained checkpoint. Every
continuation uses a new optimizer rather than resuming a previous optimizer.

| Dataset component | Train records | Validation records | Total |
|---|---:|---:|---:|
| Original 3B replay | 91,247 | 24,516 | 115,763 |
| Rank-aware win/continue labels | 210 | 45 | 255 |
| Rank-aware non-win decisions | 96 | 24 | 120 |
| Complete-menu win controls | 94 | 22 | 116 |
| **Prepared placement base** | **91,647** | **24,607** | **116,254** |

The prepared base has 48 loss-source categories in each partition. Every
prepared record was matched to its original replay or label after the declared
placement input projection. States, candidate menus, labels, soft targets and
source metadata remained exact. Late-match instructions intentionally use the
placement objective. Historical replay split metadata was preserved; its
original file partition was not reassigned from an old row-level tag.

## Independent games, score views and labels are different counts

The 491 added rank labels reuse previously acquired terminal positions from the
original 256 complete source games. They cover 152 source-family groups. The
underlying terminal collection contains 4,580 score views from 272 distinct
source-game/hand coordinates; changing scores does not create a new independent
game. All score variants retain the source family's original train/validation
assignment. The original native reconstruction audit was reused without
replaying those games again.

The fresh targeted-operation source collection is separate:

| Targeted source | Train families | Validation families | Views per family |
|---|---:|---:|---:|
| Win versus riichi | 8 | 4 | 3 |
| Win versus concealed kan | 8 | 4 | 3 |
| Win versus added kan | 8 | 4 | 3 |
| **Total** | **24** | **12** | **108 views across 36 families** |

Each family has three public score contexts: hold second, third with a 3,000
point gap to second, and fourth with a 5,000 point gap to second. The original
36 plans, complete 136-tile walls and 108 acquisition selections were checked.
These are constructed coverage cases, not a sample of natural match frequency.
Their labels are still being collected; the final 57-source dataset and its
materialized training exposures are not represented by the base table above.

## Label and source checks

The completed rank labels use a search teacher: two blocks of 16 public-world
continuations, with bounded later win replanning. This teacher uses frozen v8
continuation and public heuristic opponents. It is distinct from the Mortal
policy labels used earlier in the curriculum. Bootstrap preferences are soft
training targets, not calibrated probabilities of an optimal action.

The source join covers 2,897 previously checked ancestral assets and 661
additional base-data assets. Actual stored contents of all 661 additional files
were checked, including 491 completed paired-world cases. The review also
reconstructed 93,196 actually used later-hand native seeds from their saved
planner seeds and continuation counts. No reserved logical/native source-range
hits or unresolved source classifications were found. Deferred file-hash links
inside 484 large JSON records were checked; they introduced no missing upstream
inputs. These checks generated no new games, model predictions or training
updates, and did not recompute the bootstrap labels.

## What remains

Complete 3B training and the targeted label collection, materialize the full
placement mixture, train and audit the actual placement candidate, then extend
the source receipts to those actual final datasets and parameter identities.
The completed base-source join must not be relabelled as a final-model seal.

The predeclared final comparison remains unopened: 32 reserved logical seeds ×
four seats × two opponent profiles, for 256 complete East matches per policy and
512 total against frozen v8. Training loss and reused development curves do not
substitute for that comparison. Source preparation alone supplies no win-rate,
top-two-rate or net-point improvement claim.

This report contains aggregate counts only. Raw decisions, walls, native match
records, case-level outcomes and private experimental weights are not published.
