# Frozen v8: actual protocol ancestry and explicit-input evidence

Read-only checks followed the actual frozen reference checkpoints and training
protocols. They loaded no inference model, changed no training source, performed
no student updates and generated no reserved native game or initial-wall reset.
All work ran in finite allocated GPU-job resources.

## Actual inherited checkpoint chain

| Order | Recorded model | Checkpoint inherited by the next model, or reference |
|---|---|---:|
|1|`jev-balanced-base-lora-v2`|1536|
|2|`jev-unified-winrisk-v1`|512|
|3|`jev-unified-winrisk-v5`|2048|
|4|`jev-unified-winrisk-v6`|2048|
|5|`jev-unified-attack-v7`|2088|
|6|`jev-unified-attackrisk-v8`|2088, frozen reference|

These are actual selected local checkpoints from six original protocols,
not six stages of the new clean curriculum. Five parent edges match the saved
parent bundle identity and selected step. The six declared ancestral training
files match the actual protocol chain. Every ancestor uses the same pinned
pretrained-base revision. Original protocol input/code bindings were checked;
the inventory contains620 unique bound files.

## Explicit data and binary-input checks

| Evidence | Scope | Observed result |
|---|---:|---|
|Explicit JSONL inputs|19 files;473,071,362 bytes;203,409 nonempty lines|No reserved-seed literal occurrence; stable files|
|Explicit JSON inputs|136 files;325,423,535 bytes|No reserved-seed occurrence; stable files|
|Parameter inputs|16 adapters and16 decision heads|Hash-matched members of actual complete model bundles|
|Other explicit binary inputs|5 durable optimizer states|Weights-only schema inspection; saved optimizer counters match each state's step; no reserved-seed semantic occurrence|

The binary classification leaves zero unreviewed roles among the37 explicit
binary inputs of these protocols. Adapter/head parameters were identified by
actual complete bundle membership, not treated as raw rollout data. Optimizer
states were inspected as saved training state; they were never loaded into a
student or used to initialize this reproduction.

## Upstream producer manifests and recovered historical code

Four of the six original data roots have flat producer-input manifests. Their
union names1,280 distinct original paths with no conflicting declared hashes.
Of these,1,279 current paths match the recorded hash. One current source module,
`jevmahjong/data/bootstrap_targets.py`, differs from its historical declaration.
The original strict verification failed and its source/log/exit are preserved.

An exact archived copy matching that declaration was found and is an actual
code input bound by the original ancestor protocols. A separate explicit
historical-code mapping now verifies that one Python module against both its
original manifest declaration and that ancestral code binding. All four
manifests also match the derivation-input hashes in their actual producer
receipts. The resulting accounting is1,279 unchanged original inputs plus one
exact historical Python source. There are zero data or model-weight aliases.

The mapping preserves the original path and expected hash, separately records
the current file's different identity, and binds the exact archived bytes.
It does not overwrite the original path, manifest or failed strict check, and
does not execute the archived code. Five focused tests passed, including
rejection of data aliases, unbound/changed archives, changed manifests,
different module paths and symlink redirection. This closes the identified
historical-code lookup gap; it does not establish full source-family exclusion
or reproduce the upstream data/reward generation.

The1,279 matching upstream inputs include365 JSON,891 PT,12 Python,9 JSONL
and2 SQLite files. Hash agreement does not classify their source semantics or
replace raw-rollout/SQLite checks. In particular, this larger upstream graph
must not be confused with the37 classified direct protocol binary inputs.
Two earlier data roots have no flat manifest at that location. Their original
pool identities, selected mixtures and raw-pool derivation edges are reproduced
below. Complete recursive source-family and target-generation evidence still
requires its own closure; these checks are not a final source seal.

## Byte-exact early-mixture reproduction

Both earlier data roots were regenerated from their actual original pool
SQLite files and pool receipts, using the original seed/mixture and each
training protocol's exact archived `base_lora_mix.py`. Only output directories
changed. The numerical selection/materialization loop was unchanged.

| Original selected dataset | TRAIN records | Validation records | Actual reproduction |
|---|---:|---:|---|
|`balanced-base-lora-data-v2`|32,768|256|Both full JSONL files and selection manifest byte exact|
|`unified-winrisk-data-v1-r2`|32,768|400|Both full JSONL files and selection manifest byte exact|

This covers four full JSONL files, two complete selection manifests,65,536
TRAIN records and656 validation records. Group counts, unique-position counts
and mixture-stratum accounting also match. Original sources, manifests, pool
databases and current student inputs remained unchanged. Reproduced files stay
private and do not initialize or modify the student.

The first attempt reproduced one root, then stopped at its strict archived
module-path check because a different historical package was cached in that
process. Preserve that failure and first result. Fresh isolated processes then
ran both original loops successfully, retaining the strict import-path check.
This mixture-only check reproduces selected data from frozen pools. The
raw-pool derivations have separate evidence below. Neither check reproduces a
model-training outcome or closes every upstream source/target dependency.

## Actual ancestral raw-pool derivation reproduction

The original basic pool was rebuilt from its recorded catalog and raw sources
with the archived pool producer bound in the actual basic-model protocol.
Only the output directory changed. All79,196 retained database rows, including
every complete target payload string, match the original in canonical row
order. The source-priority file and full producer-input manifest reproduce
byte exactly, and the original strata/statistics agree.

The SQLite database file itself has a different byte hash. This result is
therefore full row/target reproduction, not a byte-identical database claim.
Both databases were compared read only and contain only TRAIN/validation
cases. Original sources/pools and the current student were unchanged; the
reproduced pool is private and never substitutes for a registered input.
The second ancestral pool's three actual derivation edges were also rerun with
the archived functions bound in its actual ancestor protocol. Each edge uses
the original hash-bound parent input and changes only its output directory.

| Actual pool stage | Full rows compared | All row/target payloads equal | SQLite file byte equal |
|---|---:|---|---|
|First basic raw pool|79,196|Yes|No|
|Second raw pool|126,628|Yes|No|
|Second pool after original validation reservation|126,569|Yes|Yes|
|Second pool after original source extension|126,744|Yes|Yes|

All three second-pool producer-input manifests reproduce byte exactly, and
all original summary statistics, strata and reservation/extension accounting
agree. These rows overlap across pool versions; they are not additive new
independent training examples. The original selected-mixture byte reproduction
continues to use its original registered pool bytes.

The first derivation-driver attempt was stopped after a duplicate progress-log
field was detected. Its original source, partial outputs and cancellation
observation were preserved; a separate corrected driver completed all three
edges with the numerical producer functions unchanged. Original inputs and
current student weights were never modified.

This closes the two identified ancestral raw-pool derivation gaps. Full
source-family exclusion, recursive upstream provenance/teacher/target evidence
and final candidate evidence remain unfinished. Reproducing the conversion of
stored raw outcomes does not reproduce the generation of those outcomes.

## Additional checks on the four-manifest upstream subset

The stored-source checks were extended to this specific upstream subset:

| Artifact type | Actual scope | Observed result |
|---|---:|---|
|Raw PT rollout sources|891 files;27,927 stored rows|All have actual raw-rollout schema; no reserved-seed semantic occurrence|
|JSON artifacts|365 files|No reserved-seed occurrence; stable files|
|SQLite catalogs/pools|2 files;754,765 examined rows|No reserved-seed literal/numeric occurrence; no unclassified BLOB cells|

The PT check reused759 previous file proofs only after exact current hash and
bound-proof checks, and newly scanned132 files. Across all891 files it covers
10,279,818,637 bytes,10,286 counterfactual windows and171,600 stored future-seed
fields. These stored-row/window totals are not independent training situations
or expert labels, and do not reproduce the selection/reward calculation.

SQLite connections were immutable and read only. The catalog inspection reads
source identity/locator metadata. The pool case-payload query is restricted to
already-produced TRAIN and validation cases; no test case payload was opened.
Literal/numeric scans do not infer opaque or otherwise encoded seed semantics.

Complete upstream source-family/target evidence and final candidate evidence
remain unfinished.
These additional checks do not establish a final source seal or model strength.

## Joined actual protocol and reproduced-producer input graph

The six actual ancestor protocols, four original data manifests and four
reproduced raw-pool manifests are now joined into one resolved inventory:
14 roots,6,329 declared edges and2,017 unique resolved artifacts. Every
resolved file hash matches its bound source evidence, including the one
explicit historical Python mapping. Seven original relative binding keys
are resolved only for lookup; original manifests and declarations remain
unchanged. The initial strict lookup failure and its original source/log/exit
are preserved, and the separate corrected inventory completed successfully.

This adds147 artifacts beyond the previous protocol/four-manifest subset.
Their actual roles and seed/source-family semantics still need review. Among
them are58 numeric wall-file paths and two archives. Header-only inspection
finds30,787 archive members:30,785 compressed `.zst` members and two directories,
with no links or unusual absolute/parent paths. Member payloads were not
opened or extracted. Ten extensionless files also have only metadata inventory
at this point. Path shapes and archive headers are not semantic classifications
or proof that the protected source distribution was unused.

The joined inventory prevents treating the earlier subset as the whole source
graph. It still does not admit the reference/future candidate to final games,
close recursive source-family provenance or reproduce the original simulation
outcomes. Original sources and the active student remain unchanged.

## What this does and does not establish

This is actual frozen-reference ancestry and explicit-input evidence. It does
not close the transitive graph of data-producer manifests, upstream raw data,
row selection, teacher provenance and reward-target generation. A matching
seed literal scan alone cannot prove the protected test distribution was
globally unused. Earlier raw-MC source checks remain separate subset evidence
until their relationship to the complete source graph is established.

The future final candidate and its completed Stage3B/Stage4 sources do not yet
exist and are not included here. Full candidate/reference provenance, a sole
qualified final candidate, the frozen registration, actual512-game comparison
and independent post-run audit remain required. No final admission, v8 parity,
production replacement or goal completion follows from this inventory.

Only this aggregate report is public. Original source paths, protocol/input
manifests, raw records, detailed findings, optimizer files, process logs and
private research weights remain private.
