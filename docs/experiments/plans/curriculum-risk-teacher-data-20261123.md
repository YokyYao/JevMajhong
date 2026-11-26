# Additional true-trajectory defensive data before the repaired third course

2026-11-23, Asia/Dubai. The basic380/608 fit and conditional RI-loss control
remain unchanged. This is data collection/audit only, with no new student,
parent admission, optimizer update or active serving/guard change.

The old third-course MC labels remain reproducible with their actual limits:
fixed opponent concealed hands,16/32 stored worlds and a one-hand continuation
with older policies. They are not public-belief/final-placement ground truth.
Do not silently replace their uncertainty or repeat the old25% MC loss recipe
as if the weak-label limitations had disappeared.

The new collection mines previously unrecorded **closed ordinary discard
choices under at least one opponent riichi**, with our own riichi undeclared.
It regenerates the same256 complete actual Mortal source trajectories used by
basic/calls data and requires exact full-event/final-score equality to their
immutable source logs. Keep the original205 TRAIN/51 validation whole-family
split across all seats/hands/phases. This adds decision windows, not independent
families. The actor sees only its actual own hand and public native events;
no sampled hypothetical hidden hands or stale private history is introduced.

Selection uses only public phase/menu/riichi/meld information and exclusion of
contexts already present in either audited basic or advanced teacher data.
Capture every eligible multi-action, all-discard window: no riichi/kan/call/win
alternatives, no existing meld, no own riichi. Exclude already recorded context
identities before inspecting the teacher's chosen discard. Do not filter by
winning outcome, safe tile, chosen action, reward, scores or a quality margin.
Source labels are the actual pinned Mortal actions, not experts' optimality
certificates. Legally available wins still follow the original source actor's
forced-win rule; this does not generate learn-to-pass-win supervision.

For every new window, record the actual current native guard's allowed-set
diagnostic without changing it. Report whether the teacher target is excluded,
whether its shanten exceeds the heuristic reference, or whether the existing
fold-risk rule excludes it. These source features and safety counts are
diagnostics, not calibrated deal-in probabilities. A legally available teacher
choice outside the serving allowed set cannot be executed by that policy even
if learned; future third-course admission must explicitly resolve or control
that compatibility issue. Do not relabel the target to the old heuristic or
disable the guard in the active run/reference to make an audit pass.

After collection, independently reconstruct every native public input/menu,
normalized ordinary Jev serving input, exact label and guard diagnostic; encode
with the pinned actual tokenizer without truncation/order changes. Verify all
basic/advanced/prepared full calls and available earlier MC/expert train/held-out
IDs/groups/input fingerprints,675 reserved old groups and whole-family splits.
Reject same-split duplicate inputs as well as cross-split leakage; do not call
old rows new data. Keep complete ledgers and exact full-trajectory matches.

The normalized new source tag is
`curriculum3:teacher:closed_discard_under_riichi`, intentionally outside the old
third-course loss adapter. No fit recipe, loss mass, stage2 parent, optimizer or
third-course budget is admitted by these files. Preserve all prepared12-source
calls replay in any later data mixture and separately audit original MC masks
and sources if they are reused. Third-course preparation cannot replace fixing
basic RI behavior or the fourth-course full-match placement teacher.

```bash
python tools/collect_curriculum_risk_teacher.py \
  --config configs/training/curriculum-risk-mortal-data-v1.json
python tools/audit_curriculum_risk_teacher.py \
  --source runs/curriculum-risk-teacher-data-v1/data \
  --out runs/curriculum-risk-teacher-data-v1/audit-v1
```

All collection/inference/hash/token/native checks/tests run in finite scheduler
GPU allocations with companion CPUs, node/GPU preflight, timeout/log/exit and
invalid-dependency cleanup. Never run on login. Publish only code/plans/aggregate
coverage and real experiment curves. New source hands/traces/teacher labels,
per-case guard diagnostics and weights remain private. Main/Windows, completed
bindings, the active basic/conditional-control chains and the unopened final
32-seed256-game-per-policy v8 confirmation stay unchanged.
