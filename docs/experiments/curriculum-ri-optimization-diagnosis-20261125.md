# RI optimization diagnosis and actual-model replay integration

The proposed replay now passes a real-model forward/backward integration batch.
The completed TRAIN gradient panel shows strong opposition between RI accept
and decline groups, while pooled new tasks are not consistently opposed to
decline. Saved predictions show a broad rise in reach-versus-best-discard
preference; reduced discard spread contributes less. Logged clipping does not
substantially change decline-versus-accept scalar coefficient ratios.

These observations narrow the next experiment; they do not establish the full
cause of the drift or prove that balanced replay improves play. The live trial
keeps its original 2,400-update recipe and advancement criteria.

## Real-model integration, no parameter update

The saved retained1200 model is an integration reference, not a selected future
parent. Its first candidate replay batch uses actual pinned augmentation,
tokenization, model encoding, forward/backward and weighted loss observation.

| Actual integration observation | Result |
|---|---:|
| TRAIN examples |73|
| Actual input tokens |60,334|
| Maximum tokens in one example |1,003|
| Source strata represented |14|
| Four inherited RI strata |Each appears once|
| Expected trainable tensors with finite gradients |376/376|
| Optimized versus observed weighted-loss error |7.96e-9|
| Preclip gradient norm |13.3459|
| Peak PyTorch allocation in this process |1.75 GiB|
| Total probe runtime |40.60 seconds|

All 372 LoRA and four pointer-head tensors match the original checkpoint both
before and after the probe. Registered fixed-base files and the reference
checkpoint were checked; all gradients were cleared. No optimizer was created,
no model parameter was updated, no checkpoint was replaced and no game ran.
This single diagnostic batch is not a learning-curve point or strength result.
Full optimizer/scheduler, checkpoint/evaluation and independent-audit integration
still needs to be registered for any future fit.

## Actual TRAIN parameter-gradient panel

The original frozen panel contains eight distinct TRAIN families per source:
152 views across all 19 sources, evaluated at parent0 and retained1200. Each
source uses the existing objective; combinations use the registered source
masses. Both models retain their exact parameters. This is an exploratory small
panel, not a full-corpus gradient estimate or native policy evaluation.

| Euclidean gradient cosine | Parent0 | Retained1200 |
|---|---:|---:|
| Decline versus pooled new tasks |+0.0999|+0.0499|
| Decline versus RI accept |−0.7876|−0.8034|
| Decline versus full combined objective |+0.1466|+0.1555|
| Decline versus RI accept, pointer head only |−0.9173|−0.9031|

The two fixed family blocks disagree on the sign of decline-versus-new-task
alignment at both models. The pooled result therefore does not support a
uniform new-task conflict diagnosis. Opposition between differently labelled
RI groups is natural and does not prove that their objectives are irreconcilable;
it makes batch composition worth testing. These are eval-mode gradients without
dropout. They do not reproduce Adam moments, head/base learning-rate ratios,
global clipping or the actual time sequence of optimizer updates.

The panel completed in 293.49 seconds. The original unstarted queue item 322921
was canceled with its real zero-runtime record preserved. Before execution,
the previously planned post-fit timing was explicitly amended after observing
about 29 GB free GPU memory and the low-memory real integration batch. The same
frozen source, samples, objectives and two model references ran inside the
existing allocation with a 20% PyTorch memory cap and 12-minute timeout; peak
allocation was 4.98 GiB. Original registration/plan and the amendment/cancellation
records remain private. Main training was not restarted or modified.

## Where saved reach probability moved

For each of the same 632 held-out RI menus at 0/600/1200, decompose the full
reach-versus-discard log probability ratio as

`full margin = reach-to-best-discard margin − discard log-sum/max gap`.

The identity holds for every inspected observation within 1e-12. A smaller gap
means the discard distribution is more concentrated. The table reports mean
changes from 0 to 1200 in log-ratio units, not probability percentages.

| Held-out source | Reach-to-best-discard change | Contribution from smaller discard gap | Full margin change |
|---|---:|---:|---:|
| Legacy decline |+0.2515|+0.0456|+0.2971|
| Legacy accept |+0.1581|+0.0450|+0.2030|
| Teacher decline |+0.2803|+0.0433|+0.3236|
| Teacher accept |+0.2871|+0.0384|+0.3255|

Thus sharpening contributes, but most mean upward movement is a real rise
relative to the best discard rather than that spread term alone. Among 14 new
legacy false accepts, holding the old gap fixed while changing the maximum
margin crosses the threshold in 13 cases; changing only the gap does so in 2.
For 44 teacher false accepts the corresponding counts are 33 and 10. These
arithmetic counterfactuals can overlap and are not independent parameter
interventions or mutually exclusive causal explanations. Validation remains
diagnostic and is not inserted into TRAIN.

Here a false accept means disagreement with a frozen held-out label; it does
not prove that every corresponding strategic choice is suboptimal.

## Logged global-clipping coefficients

All 1,200 actual first-epoch optimizer observations were clipped; the mean scalar
clip factor estimated from logged preclip norms is 0.07160. Apply the installed
clipping operation to scalar calibration gradients to recover those factors,
then account for the source weights/counts actually consumed in each update.

| Origin | Nominal decline/accept coefficient ratio | Ratio after global clipping |
|---|---:|---:|
| Legacy |1.00000|1.01658|
| Teacher |1.00002|0.99735|

This does not show substantial systematic suppression of decline relative to
accept. Both rare legacy sources have reduced normalized scalar coefficient
share relative to other tasks after clipping. Coefficient shares are not
gradient magnitudes, parameter-update contributions or Adam attributions;
directions, moments and weight decay are not recovered by this accounting.

## Next experiment and practical limit

The [every-update replay arrangement](curriculum-ri-every-update-replay-20261125.md)
is a concrete first test of update composition: all four RI strata in every
update, unchanged per-original empirical weights and about 3.01% more physical
exposures. Its [weighted loss](curriculum-ri-inverse-repeat-loss-20261125.md) and
[epoch dispatch](curriculum-ri-replay-epoch-adapter-20261125.md) already pass
complete-corpus software/data checks; actual model forward/backward now passes.
Neither reduced scheduled-weight variation nor this small gradient panel proves
that the replay arrangement will fix the drift.

Complete the live trial and its original audit. Choose any next continuation
from development evidence and explicitly register its parent, optimizer schedule,
budget, actual curves and native comparison. Keep the original quality floors;
do not infer permission to advance a failed parent or replace production v8.
There is currently no evidence requiring an extra correction network or a new
base initialization. Optimizer-path effects remain unresolved and should not
be presented as diagnosed causal facts.

No new native games or student updates were performed by these diagnostics.
The final independent grid remains unopened. Only source and aggregate findings
are published; cases, worlds, per-record weights, gradients and raw diagnostic
records remain private.
