# Replay3A after a checkpoint/evaluation ordering failure

The actual3A job318606 failed after600 optimizer updates/43800 exposures and
the full23688-row validation. The callback requested native games during the
validation tracker hook, before upstream saved its scheduled checkpoint.
Only step0 exists; no step600 model or optimizer is recoverable. The failed
source/config/metrics/zero checkpoint remain unchanged. The600 measurements
are historical failed-attempt observations, not a retained trained candidate
or600-step native performance. No600 native games completed.

Repair the execution order, not the gradient recipe. A new run directory and
ID replay from the SAME audited calls388 parent with the same seed2026112304,
data, source weights, two-epoch2400 budget, learning rates/dropout and epoch
orders. The discarded600 updates are not counted as current candidate progress.
This is a fresh optimizer from the sealed parent, not optimizer resume600.

A checkpoint fence queues post-update native work during validation and
executes it only after both model and trainer_state files exist, before the
next optimizer batch or after final fit. Step0 remains before the first update.
Save a full model/optimizer checkpoint every100 updates; validation and game
points remain0/600/1200/1800/2400.

Before the real fit, run two isolated actual upstream optimizer updates on
two TRAIN-only control fixtures with evaluation/checkpoint interval1. Verify
the real validation-before-save order, complete optimizer states at native
callback time and final callback without another batch. This is a routing
integration test, not a candidate/model-quality curve. Parent file hashes are
checked unchanged; test weights remain separate and private.

Reuse the prior REAL step0 validation predictions and sixteen native games
only after comparing all actual loaded adapter/head tensors exactly with
the preserved original zero checkpoint. Runtime/base/data are unchanged.
The new step0 protocol records this numerical-identity cache reuse. No
synthetic baseline is substituted and no existing reference game is rerun.
Post-update validation/native games still execute from actual saved weights.

The 600-step validation loss was1.198673117 (target CE1.121569601), but riichi
correct-decline recall was128/308. This cannot establish strategy improvement.
Repair does not use this intermediate result to change the fixed recipe.
Final grade and final-v8 evidence remain required; no original grade is waived.

All preparation, integration testing, hashing, model work and replay run inside
bounded scheduler GPU jobs with logs, timeouts/exits. No login compute or
unmanaged worker. No original source/checkpoint overwrite, main/Windows change,
raw-record publication or final-test opening. Research weights stay private.
## Recovery preparation correction

Recovery preparation job 318957 exited before candidate training: the two-update control fixture incorrectly reused identical train/evaluation rows and upstream correctly rejected semantic overlap. The failed v2 source, log and exit receipt remain unchanged. Recovery v3 uses four semantically distinct TRAIN-only control examples, split into two update rows and two held-out control rows; these are only a save-order integration check and never a model-quality validation result. The main recipe, data, seed, fixed2400 budget, retained stage1/2 parent and final advancement criteria remain unchanged. The candidate output is isolated at `runs/curriculum-progression-v3`; discarded failed600 updates are not part of its effective ancestry.

