# Explicit Jev label adapter after a pre-update encoding failure

Fit309763 exited1 on the first training record before any optimizer update or
checkpoint save. The pinned Jev `materialize()` requires `question.label` even
when `question.target` exists. The frozen teacher records contain the mapped
action in `_meta.teacher_action` but omitted the duplicate question.label field.
Interface unit tests had not exercised the actual upstream materializer.

Preserve original sources,1898/256 data files, targets, quotas, split identities,
their hashes and the original generator registration. Convert a COPY of each
teacher row at the explicit encoder boundary: label=_meta.teacher_action, with
legal membership and any existing label agreement required. Existing retention
labels remain required; no inference, new target or action choice is invented.
Reject conflicting labels, missing retention labels and duplicate identities.
The same transformation is used by actual fitting and by the GPU schema check.

Before loading the student model, materialize ALL1898 actual records with pinned
upstream Jev. Record this check separately. Archive failed metadata/provenance
from309763, then start a fresh optimizer from EXACT176 again; this is not a
continuation of nonexistent updates. Training/selection/evaluation seeds,
loss masses and the complete-match diagnostic remain unchanged.

Re-run changed trainer tests and the actual full-record materialization in an
allocated GPU job, freeze the corrected source, then launch bounded fitting,
normal-load audit and full-match comparison. Never overwrite failed receipts or
claim the earlier launch produced trained weights. No login computation.
