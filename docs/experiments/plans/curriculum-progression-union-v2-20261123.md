# Exact union v2: preserve historical source-file partitions

GPU317986 on ws-l4-005 fails1:0 in1m27s while checking the first inherited
record: old public records have no per-row `split` field. Their partition is
already defined by their hash-bound TRAIN/validation source files and original
whole-family holdout. This is a preparation schema assumption, not a model or
label-quality result. Its sourcecc28173f7560e45a636faf38694d4695d17a2307,
failed output directory, tests, log and actual terminal/shell exits stay intact.

V2 uses new helper/config/test/scheduler entry and a new `data-v2` directory.
An inherited record without that optional field keeps its exact source object;
no `split` is inserted. Any supplied split must still agree with the bound file.
Mortal records carrying source logical seeds must have their explicit split.
The full whole-family, reserved-group and cross-split public-input/ID exclusions
remain required. Count inherited records using source-file partition authority.
The producer's focused test adds this distinction; no failed check is skipped.

All other [v1 requirements](curriculum-progression-union-20261123.md) remain:
exact calls replay49,022/12,479, risk6,860/1,948, ordinary28,991/8,077,
informative conditional MC2,722/1,184, total87,595/23,688; original structural
singleton quarantine136/41 separately. Preserve observed labels, masked support,
soft targets and metadata, recheck the existing TRAIN-world evidence, and never
open validation rollout payloads. Report same-split duplicates; fail cross-split
leakage. No new model, label, simulation, loss mix, parent or fit is admitted.

Run only the finite GPU scheduler entry with node/device checks, logs/timeouts/
exit. Recheck224 active-fit dependencies unchanged. Source and aggregates may
be public; raw records and new weights stay private. Independent union/encoder
audit and the actual completed-parent execution comparison are still pending.
