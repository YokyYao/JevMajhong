# Complete placement data origins and incremental metadata

The placement fit remains downstream of the complete original 108-view label
collection and its original preparation and exposure recipe. These utilities
do not change training labels, sampling weights, validation partitions, final
acceptance criteria, or checkpoint selection.

`tools/prove_curriculum_placement_complete_data_origins_v1.py` requires the real
completed exposure corpus. It compares every current TRAIN and validation row
with the actual closed base or complete targeted parent, using the frozen
serving projection. It preserves candidate insertion order as well as values.
It then restores every virtual TRAIN copy and checks that only the generated
ID and explicit exposure trace changed. Validation must be reused unchanged;
an inherited `_meta.split` does not override the actual current file partition.

Four boundary checks and actual two-row frozen-base projection parity passed
in allocated GPU job step `326753.1`. The complete-data proof job `326787` is
queued after the genuine exposure producer, with terminal capture `326788`.
The full proof has **not run yet**. It is independent of training admission.

`tools/observe_curriculum_metadata_delta_v1.py` consumes a completed bounded
metadata inventory and observes only its new JSON files. It checks decoded
numeric values, dictionary keys and numeric strings for both reserved logical
seeds and their spaced native hand streams. An exact protected-range
declaration does not exempt a separate data occurrence in the same file.
Large inputs and other types require their own observer.

GPU job step `326753.2` completed four boundary tests and observed the three new
3B completion/audit/terminal metadata files, totaling **29,490 bytes**, with no
unreviewed reserved occurrences. It did not rescan closed rollout payloads,
load a model, generate games, or update weights. The preceding inventory had
already found the unchanged 3B protocol in the closed source catalogue.

Both outputs are source evidence with explicit limited scope. Neither is the
complete final-candidate source proof, a final test result, or authorization
to open reserved games. A real qualified Stage4 candidate, its actual complete
training ancestry and the full targeted-source observations are still needed
before the separately registered final comparison against frozen v8.

Only reusable code and this aggregate record are public. Raw hands, simulated
worlds, local paths from execution receipts and research weights remain private.
