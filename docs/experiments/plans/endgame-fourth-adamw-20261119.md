# Fourth-place pursuit: longer full-source AdamW ablation

The user asked why round16 stopped after two updates and authorized analysis,
revised training and continued GPU work. Two evaluations immediately after two
very small updates are not evidence of convergence. This run tests that training
control hypothesis before changing the encoder, labels or loss masses.

## Evidence and limits

Round16 allowed only eight updates and stopped after two stale validations,
one validation per update. Every source CE had to decrease at every update.
The initial relative head step .005 was halved four times on both updates,
to .0003125 (0.03125% of current parameter L2 norm). All eight source CEs
decreased, but fourth validation soft top2 moved .18299680066 to .18299088686;
hard top2 did not change. Training strict target hits moved 14 to15/48.
This establishes a weak, short update trajectory, not an optimum or proof that
additional training cannot work. Cross-source conflicts and conservative line
search are hypotheses to inspect with measured source gradients/losses.

Some targets are uncertain: the prior training-only paired-world audit found
5/19 ordinary-fourth strict targets remained strictly beneficial under every
leave-one-world-out check. This does not classify the other14 as wrong or
justify filtering rare fourth-to-top2 opportunities. All data remain.

The frozen-v8 fourth-only endgame path also failed to improve top2 on consumed
complete continuations (7/33→7/33). Thus changing the decision path alone did
not deliver an observed gain; the static51-menu signal was not sufficient.

## Registered training

- Start again from released Play v2/v8 step2088, not a rejected checkpoint.
  Same original524,800-parameter pointer head; backbone and rank16 LoRA frozen.
  No extra inference network. Existing exact v8 encoder caches are valid.
- Reuse round16 unchanged905 train/186 validation memberships, all826 positive
  loss rows on EVERY optimizer update. Fourth supervision149 rows; fourth
  validation51. All source masses/targets/references unchanged. Fourth total
  loss mass .35 (.20 ordinary,.15 win/continue); original/leading replay .50;
  third/holding .15. No train/validation reshuffling or label generation.
- Replace projected every-source monotonic descent and line search with
  full-source AdamW: fixed mean source gradients weighted by the same masses,
  LR5e-5, linear warmup8 updates then constant, weight decay.001, norm clip1.
  Per-source CEs may temporarily increase. Log all source CEs, gradient pair
  cosines, every update norm and cumulative parameter distance.
- At most128 full-data updates. Validation/checkpoint every8, including zero.
  First32 updates are an observation period. Only validations AFTER32 count
  toward four-check no-progress patience; with no progress, earliest ordinary
  stopping is64. Progress means new hard fourth top2 best, eligible selection,
  or soft fourth top2 improvement >1e-5. Soft gains only extend observation;
  they cannot qualify weights. Nonfinite training aborts; original replay CE
  >zero+.50 at a validation stops for severe regression, separately reported.
- Selection is EXACTLY round16 `chase_guards`: actual fourth top2 must exceed
  released reference and input-path zero, with original third/holding/replay
  guards retained. Fourth escape and native-fourth KL are report-only. A loss
  decline, model confidence gain or training target hit alone cannot promote.

## Analysis and next stages

GPU artifacts retain initial training gradient conflicts, per-update full
source losses, full186 validation predictions and149 fourth training choices.
Compare step2 vs later update size/training hits and held-out soft/hard top2.
This is an optimizer-plus-observation-window ablation; it does not identify
which one alone caused any change. No validation gradient trains the model.

If a nonzero checkpoint qualifies, audit actual ordinary model loading at the
selected and final checkpoints and check tensors/probabilities, then register
a small consumed full-continuation development comparison with the exact
candidate policy. Fresh independent/full-start sources remain untouched until
that succeeds. If the head learns training targets but not validation,
additional epochs are not established as useful: investigate target support,
coverage and representation changes (LoRA), rather than automatic repetition.

All computation, hashes, tests and diagnostics run inside bounded Slurm GPU
jobs with allocated companion CPUs, logs, exit receipts and managed children.
Public output is code, this plan and aggregate findings; raw data, features and
research weights are private. Main, released model and Windows0.4.2 stay as-is.
