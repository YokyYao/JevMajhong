# Prospective 3A control: retain the qualified parent's replay objective

The paired-order3A retry completed its2400 updates,23688 validation decisions
and64 native development matches. Final validation still fails the original
RI-decline requirements (legacy40/70; teacher127/238). Native top-two rate is
48.4375% and net points are+1087.50/match, versus same-grid frozen-v8's59.375%
and+4290.625. Its independent execution audit is still running when this plan
is written. The failed final cannot admit the existing3B/placement chain.

Cached prediction reconstruction shows a second late upward RI shift:
1800→2400 changes37 decisions toward RI and two toward decline. The18 newly
correct accepts come with19 newly incorrect accepts and two corrected false
accepts. Pooled class-score AUC rises from0.76409 to0.76764 while mean RI
inclination rises for both label classes. These are consumed-label diagnostics,
not causal proof or Mahjong action values. The paired ordering did not resolve
the final retention failure or improve the native point estimates.

There is a concrete objective difference to isolate. The qualified calls
parent retains, for basic RI rows, a binary loss against the largest discard
logit plus the conditional discard loss. Existing3A replaces it with twice a
binary loss against log-sum-exp of all discard logits plus the conditional
discard loss. That changes both the comparator and the relative gradient
pressure. Ordinary replay and advanced call losses also need exact semantic
retention rather than an accidental redefinition during the course transfer.
The observed probability shift motivates this control but does not prove that
the objective difference is the sole cause of lower complete-game scores.

The proposed control keeps the exact original calls13 objective on every
inherited replay source by invoking its existing annotation/loss functions.
The two new hard-teacher sources and four masked-MC sources keep their current
progression objective. Source loss masses, raw records, target masks and
inference policy stay unchanged. This introduces no residual network, new
teacher, probability threshold search or post-hoc label replacement.

RI maximum-comparator loss need not equal ordinary target CE. Its difference
from CE can be negative on positive RI examples. Record that quantity as a
signed diagnostic, alongside the actual maximum-comparator and conditional
terms; do not call it a nonnegative extra RI penalty or reuse the old Class2
objective identity. Validation must reconstruct the new actual objective.
The CE, source confusion matrices and native game definitions remain comparable.

The intended actual student starts again from the same qualified calls388
parent with a fresh optimizer. Both failed3A fits remain separate controls;
their updates are excluded from its parameter ancestry. The numerical budget
remains two epochs/175190 exposures/2400 updates, batch1/accumulation73,
LR3e-6/head1.5e-5, the paired511-record block order and the same consumed
validation/native grids. Only the replay objective changes from the paired
control. Retain the original advancement thresholds and the sole final-budget
candidate; do not promote a process high point. Conditional3B/placement entries
will need explicit new qualified-parent registration after the actual result.

This plan and objective module do not admit or start a fit. Required launch
work remains: finish/preserve the actual control audit, freeze a separate
registration and outputs, exercise the objective on the actual TRAIN source
inventory, verify the full validation reconstruction, prove exact parent
tensor/prediction inheritance, bind the trainer/evaluator/auditor implementation
and submit finite GPU jobs with real logs and exit records. No current or bound
historical source is modified. Supplemental kan-source labeling continues
independently and is not silently added to this controlled recipe. The final
reserved32-seed comparison remains unopened, and no production model changes.

Objective entry: `tools/curriculum_progression_parent_loss_v1.py`.
Its tests compare exact loss values and gradients for all13 replay sources,
the six remaining sources, signed accounting and immutable request fields.
