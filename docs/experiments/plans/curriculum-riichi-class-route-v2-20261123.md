# RI route execution repair, preserving the frozen comparison

The [v1 policy protocol](curriculum-riichi-class-route-20261123.md) remains the
scientific comparison. GPU316433 fails1:0 after30s during original-policy
baseline reproduction: `write()` uses exclusive creation for every file, so
the second update to `status.json` raises FileExistsError. The failed directory,
sources, preflight239 checks and full logs/bindings remain immutable. Dependent
audit316434 is cancelled before starting. No candidate64 result is available.
Neither the already completed608 training nor its audited64 result is rerun.

The v2 wrapper changes only status-file persistence: atomically replace status
via a temporary file and retain exclusive creation for every other artifact.
Its temporary function bindings are restored on normal exit or exception.
New tests reproduce successive status updates, preserve summary overwrite
rejection and check dispatch restoration. The preflight also checks the actual
partial baseline records against the immutable original cache and confirms no
candidate-match file was created. Use fresh v2 output directories; never append
to or hide the failed v1 attempt.

Threshold0.5, policy/source weights, source seed grid, safety guard, no-forward
win shortcuts,16 exact baseline matches and64 candidate matches are identical.
No threshold is tuned, model trained or inference behavior repaired by this
change. Reuse v1 selection/evaluation/audit under the scoped wrapper, bind the
new wrapper/config/tests/protocol, and require fresh GPU preflight and source
CI before a new managed trial plus independent audit. Keep actual failures
and operational receipts alongside any eventual result.

All computation remains in finite GPU scheduler jobs with companion CPUs,
resource checks, timeout/log/exit and cancelled invalid dependencies. No login
compute, CPU-only workaround, unmanaged worker, parent admission, production
replacement or unopened-final-grid use is introduced. Full four-course/v8
objective remains active and unachieved.
