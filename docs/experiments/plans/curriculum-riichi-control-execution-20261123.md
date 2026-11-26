# Executable conditional fresh RI-loss control

2026-11-23, Asia/Dubai. This implements the already stated
[single-factor plan](curriculum-riichi-class-loss-control-20261122.md). It does
not change or shorten the active hard-action-CE608 fit. No control training may
start from an unfavorable midpoint. The launch condition is an actually
completed608/final64 integrity audit with recomputed source-quality failure.
If current basic passes all sources, skip this experiment and continue the
qualified curriculum. Failure is not admission of that model as a parent.

Freeze `curriculum-riichi-class-control-v1.json` against the current registered
control: fixed Qwen revision/base snapshot, fresh rank16/head256, same seed,
all14,285/3,266 inputs/labels/groups, loss source weights, accumulation94,
four full epochs608/57,140 exposures, LR/head LR/decay/dropout and serving/guards.
All actual trainer arguments except the new output must match. Candidate rule
remains registered final full budget. Only the RI binary loss component differs.

The registrar reuses the fixed-base/data validation and preserves its initial
generic receipt before freezing the new explicit loss/source bindings. It
does not fit a model. The new trainer wraps only the pinned materializer's
training-only RI indices, loss/metric callbacks and ordinary metric collection;
all hooks restore on exit. Public inputs and the normal serving model remain
unchanged. Changed objective, original hard-action CE and all six component
means/counts are logged separately. Actual objective reconstruction uses
`source_objective`; `source_ce` retains its original hard-action meaning.
Validation stores both common CE and the changed objective, with original
plain/legacy/teacher quality grades preserved.

Before the first optimizer update, compare every372 adapter/four head tensor
with the original fresh control's saved0 (comparison only, no weight loading),
all3,266 initial validation distributions and all16 matched initial outcomes.
They must reproduce the original within the same probability tolerance5e-5
and exactly for tensors/game outcomes. The ordinary frozen-base/fresh-LoRA
audit and actual raw-input update proof remain required. Four actual epoch
order receipts must equal the original control's, not just have equal counts.

The independent auditor checks all608 actual updates/57,140 exposures, all376
optimizer states, each source objective/common CE decomposition, all validation
profiles, initial equality and real serving reloads on all3,266 menus at0 and
608. It verifies complete-game ledgers/identities and real curve provenance,
then reports whole-seed paired64-match deltas to completed hard-CE608 and
frozen v8. The original source-specific quality gate stays unchanged. Lower
modified loss or a small matched16 peak is not success; full later courses and
the untouched final256-match-per-policy comparison remain the actual goal.

An actual launch receipt must bind the fixed registration, successful public
source CI, original final audit/completion/validation, actual successful
allocated-GPU fit/audit scheduler archives and original final64 cache. The
fresh control trainer and its auditor recheck the trigger instead of trusting
a status label. No fabricated terminal output, failed-parent warm start,
minimum-CE selection or hidden change in source weights is permitted.

```bash
python tools/register_curriculum_riichi_control.py \
  --config configs/training/curriculum-riichi-class-control-v1.json \
  --base-path "$FIXED_BASE_SNAPSHOT" --reference-path "$FROZEN_V8"
python tools/train_curriculum_riichi_control.py \
  --registered "$CONTROL_REGISTERED" --launch-receipt "$ACTUAL_FAILURE_LAUNCH_RECEIPT"
python tools/audit_curriculum_riichi_control.py \
  --registered "$CONTROL_REGISTERED" --launch-receipt "$ACTUAL_FAILURE_LAUNCH_RECEIPT" \
  --out "$FRESH_CONTROL_EVIDENCE" --binding "$CONTROL_LAUNCH_BINDING"
```

Run only in bounded scheduler GPU jobs, including registration/tests/hash/data
checks, with actual node/GPU preflight, companion CPUs, timeout/log/exit receipt.
The managed fit is scheduled only after the actual current audit and terminal
archive task complete; invalid dependencies must cancel rather than remain
unmanaged. If a prerequisite fails, inspect the same actual handles and fix the
specific failure; never silently restart a partial output. Publish source/plans/
aggregate results/real curves only. New raw data/derived weights/predictions and
research weights stay private. Main/Windows and the final test grid stay intact.
