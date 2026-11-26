# Complete legal menu with the fixed RI class rule: execution preflight

This prospective execution route removes the old shanten/risk filter from
action selection. The student can select every native legal discard, including
ordinary no-RI draws, open-hand followups, and discards offered alongside kan.
The filter remains a shadow diagnostic. Legal RON/TSUMO shortcuts and single
action shortcuts stay unchanged. Exactly one model call is made for any other
decision. Only menus whose types are exactly reach/dahai use the existing full
RI probability threshold0.5; mixed kan menus keep their existing argmax rule.

The current388-step calls training and production are unchanged. The frozen v8
reference keeps its original execution route. A route change has its own policy
identity and cannot inherit the old route's game results or event equivalence.

## Why this is needed

The independently audited ordinary completion has37,068 direct Mortal source
discards. The old filter excludes418 TRAIN and93 VAL labels despite all being
legal. Those have no RI or melds, so the earlier threat-only risk scope cannot
fix them. Native legality does not imply strategic quality; exposing choices is
an execution correction, not evidence that the resulting policy is stronger.

## Registered bounded GPU work

Run `cluster/curriculum-complete-legal-class-v1.slurm` only in a scheduler GPU
allocation. Verify the assigned node/device, three focused checks, all immutable
active-fit dependencies, and the pinned runtime. Read the five independently
audited teacher cohorts (basic, advanced, risk, post-reach, ordinary completion),
require82,419 distinct decision IDs with their original splits, verify their
audit bindings and data hashes, and test synthetic target-preferred distributions
over every stored menu. These are interface probes, not model predictions or new
teacher labels. Do not infer the unclassified21 source network windows.

Select a bounded native subset deterministically: first row for each cohort,
split, labelled action type, and saved guard reason, plus the first row offering
each action type per cohort/split. Reconstruct every selected physical source
prefix and exact normalized public menu. For each candidate, supply a synthetic
0.95 preference and require the actual agent to return that native action with
one call and an unchanged request. Check0.4/0.5/0.6 RI class boundaries against
the old route. Execute separate native RON and TSUMO shortcut fixtures with zero
model calls. The scripted walls are integration controls, not training data.

Keep raw rows, source IDs, traces, and per-case outputs private. Publish only
code, this plan, and aggregate results after actual success/terminal evidence.
Write a new output directory, logs, bindings and shell exit; bound runtime to
one hour and preserve failures unchanged. No new student weights are loaded.

## Next admission requirements

After the actual calls388 parent and independent audit finish, compare the same
completed parameter bundle on the consumed native development grid with the old
and proposed routes. Do not open final confirmation seeds. Actual warm-start
tensor/probability equivalence is still required for a later fit, while a changed
route's step-zero events may legitimately differ. Only then register the merged
data, loss mix, update cadence, and risk/progression fit. This preflight does not
admit that fit or establish parity with v8.
