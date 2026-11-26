# Conditional single-factor RI class-loss control, not a launched fit

2026-11-22, before basic608 final results. The active hard-action-CE run stays
unchanged. This plan records one specific control justified for preparation by
the consumed0/76/152/228 diagnosis; it does not claim that its loss will improve
policy strength. Do not launch solely because a midpoint is weak. First finish
the current608/final64 audit. If it still fails basic/RI quality, complete this
control's registration/entry/full preflight and public-source CI before any
student update. If the current recipe succeeds, continue the qualified courses
instead of automatically spending another full run on this proposal.

Keep fixed Qwen revision, fresh LoRA/head initialization and seed2026112002,
all14,285/3,266 inputs/labels/groups, six source weights, complete94-record
sampler/152 updates per epoch, LR2e-5/head1e-4, decay/dropout, four complete
epochs608/57,140 exposures, serving/guards and consumed development game grids.
No old Mahjong/Mortal weights, data additions/removals, source reweighting,
extra network, decoder rule or new threshold changes accompany this control.
Original source-quality grades and final-v8 objective remain unchanged.

Only replace the RI class component of the loss. For a riichi menu with one
reach logit `r`, ordinary-discard logits `d`, and the actual hard class `y`:

```text
existing class CE = softplus(logsumexp(d) - r)   when y=reach
                  softplus(r - logsumexp(d))   when y=discard
proposed class CE = softplus(max(d) - r)         when y=reach
                  softplus(r - max(d))         when y=discard
conditional CE   = 0                           when y=reach
                  CE(d, actual_discard_label)  when y=discard
```

The existing action CE decomposes as existing-class CE+conditional CE. The
prototype substitutes proposed-class CE with coefficient1, retaining the
conditional term and unchanged per-example source multiplier. Plain rows keep
their exact hard-action CE. The new class margin matches the unmodified
individual-action argmax decision boundary; `amax` distributes gradients over
tied maximum discards. This is a mathematical alignment hypothesis, not a
causal explanation for all observed drift or a guarantee of optimal RI labels.
The legacy107/teacher525 validation windows remain development diagnostics.

The pinned materializer already retains actual candidate keys/order and labels.
After it runs, attach only training-only reach indices computed from the same
raw menu. Reject missing/multiple reach options, inconsistent labels or altered
keys. Removing these private loss indices must reproduce every materialized
input unchanged; the raw request, model input and serving API remain identical.
No hidden hands, future outcomes or validation labels enter a training request.

Log actual objective components, ordinary hard CE and the original class CE
separately for every source/update. Use `source_objective` to reconstruct the
training loss; never relabel the changed objective as the old action CE. Keep
common validation action CE, actual modified validation objective, per-source
accept/refusal, plain retention, AUC and all actual saved-checkpoint reward,
win/top-two/net/deal-in curves. Initial tensors/predictions/same16 outcomes must
match the existing fresh control exactly before any updates. Choose only the
fixed608 final, independently reload/audit it and compare full64 development
matches, then proceed through all remaining courses only if qualified. A
favorable midpoint, lower modified loss or passing a prototype gradient test is
not a final student or v8-strength result.

`tools/curriculum_riichi_class_loss.py` is a prototype for this conditional
control, with tests for CE decomposition, unchanged plain/two-action loss,
permutation/shift/tie gradients and private-index/input preservation. Actual
Qwen training/registration, real complete-run execution and strength evidence
are still pending; do not present logit-gradient checks as a trained policy.
All compute/tests require bounded scheduler GPU resources; new raw examples,
derived weights/predictions and research weights stay private. Main/Windows
and the unopened final32-seed confirmation grid are unchanged.
