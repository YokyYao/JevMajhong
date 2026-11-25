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

## Upstream producer-manifest gap

Four of the six original data roots have flat producer-input manifests. Their
union names1,280 distinct original paths with no conflicting declared hashes.
Of these,1,279 current paths match the recorded hash. One current source module,
`jevmahjong/data/bootstrap_targets.py`, differs from its historical declaration.
The original strict verification failed and its source/log/exit are preserved.

An exact archived copy matching that declaration was found. This is recovery
evidence, not a replacement of the original path or proof that the whole graph
has passed. Neither the original manifest nor current code was rewritten.
Admission of any explicit historical source mapping remains unfinished.

The1,279 matching upstream inputs include365 JSON,891 PT,12 Python,9 JSONL
and2 SQLite files. Hash agreement does not classify their source semantics or
replace raw-rollout/SQLite checks. In particular, this larger upstream graph
must not be confused with the37 classified direct protocol binary inputs.
Two earlier data roots have no flat manifest at that location and still require
their original pool receipts and specialized producer evidence.

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
