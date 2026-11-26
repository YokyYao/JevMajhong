# Preserve the paired retry's actual terminal records

The original launch-bound capture source is retained. A file-I/O integration
check reproduced its attempt to exclusively create the same observation path
twice, once per role. Its pending task was canceled before execution. This
version uses separate exclusive fit and audit observation paths; a complete
two-role integration exercises the actual capture path and keeps failed strategy
grades unchanged. The training and independent auditor are unaffected.

The controlled 3A retry keeps its registered training and independent audit unchanged.
A separate, finite GPU scheduler task runs after the audit terminates and copies
the original scheduler terminal observations and exit receipts before records expire.
Its source, template, checks and exact parent registration are bound while the fit
is actually running and before its final update. No polling service or login
computation is introduced.

Successful fit evidence may reuse the original independent auditor's exact archived
fit terminal when the scheduler has already retired that record. Audit terminal
evidence must be observed from the actual scheduler. A missing observation is a
missing proof; no record is reconstructed from filenames or expected outcomes.
Failed, canceled and timed-out jobs are preserved as unsuccessful outcomes.

This task never admits a child, changes a strategy grade, selects a checkpoint,
loads a model, opens evaluation seeds or starts another student fit. Passing
execution and passing the unchanged strategy advancement conditions remain
separate requirements for any later 3B parent registration. Scheduler records,
resource details and exit receipts remain private.

Entry: `tools/capture_curriculum_paired_terminal_v2.py` (`bind`, then `capture`).
Template: `cluster/curriculum-paired-terminal-capture-v2.slurm`.
Checks: `tests/core/test_curriculum_paired_terminal_capture_v2.py`.
