# Continue the128-step pursuit LoRA trajectory through384

The user requested more training or additional data after a small validation
gain. First isolate duration: continue the final128-step state chronologically,
including optimizer moments, source sampler position and Torch/CUDA RNG. Do
not restart from v8, rewind to the best-looking96-step checkpoint, reset
warmup or mix a data change into this experiment.

## Why continue, and what remains uncertain

The previous128-step run reached its registered maximum, not early stopping.
Its late soft fourth validation values kept increasing: .18823 at80,
.18909 at96, .19154 at112, .19214 at128 in live unmerged evaluation. Greedy
finite-label top2 peaked .230392 at96, then .225490 at128 versus formal
same-path zero .223039. Actual normal serving at128 confirmed .225490, and
third/holding terminal objectives did not decline. This supports testing a
longer trajectory; it does not establish convergence or real match gains.

The old native-third KL .14536 exceeded .05 despite unchanged third terminal
utility. A preference-distribution distance is not itself a measured ranking
loss. For this NEW research run, record historical guards exactly as before,
but screen candidates by terminal outcomes. This does not retrospectively
qualify round18 under its old rules or authorize a release.

## Frozen protocol before new updates

- Prepare with actual normal-service checks of BOTH parent96 and128,335
  requests each. Freeze the stronger of their fourth hard top2 values and
  formal zero as the extension's improvement floor. Thus a result worse than
  the already available parent peak is not advertised as a new training gain.
- Resume128 LoRA/head tensors exactly, AdamW moments/step128, original rates
  (LoRA2e-6,head1e-5), decay.001, clip1 and RNG. Verify expanded384-batch
  schedule's first128 batches equal the original protocol. No repeat warmup.
- Same905 train/186 validation memberships,826 positive rows,149 fourth
  supervision menus,51 fourth validation menus, source masses and32-record
  balanced batches. No new labels or simulations. Only tokens are reused;
  encoder activations and LoRA/head gradients are computed anew.
- Add at most256 updates, reaching total384. Full validation every32. Observe
  through total192; afterward three stale checks stop (earliest288 if no
  hard/soft progress). Soft fourth top2 improvement>1e-5 extends patience but
  does not qualify a candidate. Nonfinite loss/gradients abort; replay CE
  >formal zero+.50 stops separately. Save every evaluated adapter/head,
  optimizer, RNG, probabilities and exposure with global step numbers.
- Research screening: hard fourth top2 must exceed the best audited parent;
  ordinary-third and overall third terminal utility cannot drop below formal
  zero, holding utility may drop at most.02, no severe replay CE regression.
  All KL distances, ordinary expert CE and fourth escape remain visible.
  Historical formal guard decisions are reported unchanged alongside this
  research screen. Screening never marks weights release-qualified.
- After training, automatically load the screened selected and final weights
  through ordinary serving, remeasure the same metrics, and require actual
  service to pass the research screen before a consumed-development candidate
  can be considered. Differences from unmerged training are not assumed zero.

## Stages and data expansion

Track total192,256 and384 checkpoints. If training targets improve but held-out
hard top2 stalls or actual third/holding declines, do not extrapolate soft
progress into a strength claim. Next consider NEW TRAIN source groups with
varied deficits/remaining hands and both beneficial and harmful chi/pon/kan
and small-win/continue choices. Repeating existing records is not new data.
That expansion needs its own source-isolation/label-budget protocol; none is
claimed in this duration experiment. Existing validation remains fixed.

Only after a saved candidate clears research screening, register a small
complete-continuation comparison on already consumed development sources to
test actual top2/third/holding outcomes. No fresh independent/full-start pools
or desktop/main replacement are automatic. The51 finite-label menus are not
complete-match win/top2 rates. Additional duration may still overfit them.

All computation/hashing/tests run inside bounded Slurm GPU allocations with
companion CPUs, foreground children, logs and exit receipts. Only code, plans
and aggregate findings are public. Raw traces, features and research weights
stay private. Formalv8/main/Windows0.4.2 remain unchanged.
