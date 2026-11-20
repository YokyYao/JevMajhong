# Fresh-base lineage: endgame-retention progress

Status: GPU fit310255 is active; saved-model validation has reached32 new updates. No retention checkpoint is selected yet, and no retention candidate has received complete-game evaluation. The [registered plan](../../plans/mortal-base-retention-20261120.md) requires at least128 new updates and all2,096 admitted examples exposed before selection/stopping; maximum384, validation every32 and four stale checks thereafter.

## Parent and actual initialization

The [foundation experiment](mortal-base-progress.md) stopped at1408 and selected1152. Its64-game result was24/64 top-two (37.50%) and-2,350 points/game, versus31/64 (48.44%) and-1,445.31 for the audited176 reference. It did not demonstrate improvement. The present bounded follow-up keeps the registered1152 parent to measure the effect of terminal retention; it does not claim1152 is the strongest playing checkpoint or select a parent post-hoc from game scores.

Foundation fit309978, final evidence audit309982 and retention preparation310254 completed with controller state `COMPLETED`, exit `0:0`, matching their exit receipts. The new fit verified exact1152 adapter/head tensors, a new optimizer, frozen backbone,10,822,656 trainable LoRA parameters and524,800 head parameters. No old v8/128/176 policy parameters or optimizer were loaded, and no runtime network was added. The stage0 saved normal loader reproduced every foundation teacher-validation metric exactly.

All step counts below are **new updates from selected1152**. For example, new32 has checkpoint index1184 in this separate run. The discarded foundation updates1153–1408 are not its initialization; matching numeric indices across experiment folders do not imply matching weights.

## Data and objective

Training contains2,048 outcome-blind teacher TRAIN replay decisions plus48 terminal-outcome TRAIN positions. Validation contains the same512 separate teacher decisions plus8 separate terminal monitor positions. Whole source-game groups and decision IDs remain separated. The foundation had already exposed all22,321 teacher training decisions; this replay subset regularizes the follow-up rather than replacing that foundation history.

The loss uses85% population-calibrated teacher mass and15% terminal-outcome mass. Endgame targets are recomputed from the fresh parent's own distribution and existing paired64 binary final-top-two outcomes, not copied old priors/soft targets or Mortal choices. Those outcomes still depend on old continuation policies and fixed original hidden hands; they are not expert labels or a public-state posterior. No fourth-escape non-decrease veto is applied.

An actual-agent replay found5 held-position records with extra display context. The first pending successor jobs were canceled before starting; no data/updates from that revision were admitted. Corrected inputs now exactly match56/56 independently captured native requests, while preserving core public state and ordered candidate features. Five checks and both core/desktop CI44650768159 passed for source `a037dd44ea343aaf3c827c48691f0fc44e86a864`; matching source bytes and private post-fit scripts were verified/frozen in the GPU allocation. The admitted dataset verifies all56 actual native inputs.

## Saved normal-loader checks

| New updates | Teacher weighted CE ↓,512 menus | Terminal-target CE ↓,8 monitor menus | Joint CE ↓ | Distinct TRAIN examples exposed | Selection |
|---|---:|---:|---:|---:|---|
| 0 | 0.75664 | 0.68085 | 0.74527 | 0 | Observation floor not met |
| 32 | 0.75828 | 0.69757 | 0.74917 | 1,328/2,096 | Observation floor not met |

At32, both LoRA and head have changed, but joint loss is slightly above the baseline. Exact teacher-positive/negative call matches move40/64 and63/64 →39/64 and63/64. Riichi has27 teacher-positive action-type matches with17 extras. No positive daiminkan teacher supervision has appeared. These are auxiliary teacher matches, not independently correct action labels or win rates.

## Cached terminal monitor, not complete-match top-two rate

| Initial placement stratum | Monitor positions | Parent new0 cached top-two estimate | New32 cached top-two estimate |
|---|---:|---:|---:|
| Fourth | 6 | 24.48% | 24.48% |
| Third | 1 | 46.88% | 46.88% |
| First/second holding | 1 | 43.75% | 43.75% |

These values apply the model's chosen action to stored counterfactual outcomes, with old continuation behavior after that action. They are not freshly played games, an independent test set or the model's overall top-two rate. New32 shows no change in these hard estimates; the tiny8-position monitor cannot establish general playing strength. No benefit is claimed from the small soft-distribution changes. Continue the registered128-update/full-exposure observation period rather than stopping after an early small number of updates.

## Remaining evaluation

After fitting, reload selected/final checkpoints and reproduce their metrics, then run64 new complete East games for the selected retention candidate. Reuse both the audited176 reference games and the fresh1152 foundation games; rerun neither baseline policy. Compare against both with whole-seed uncertainty and independent payment/pool/rank reconciliation. Dependency-bound GPU audit310256 also checks real optimizer states/exposure, exact parent lineage, all56 normalized inputs/targets and source/runtime identities. These reused development comparisons are not independent confirmation.

Training and final evidence remain incomplete. Raw records, detailed predictions and weights stay private; no automatic model/EXE release or main merge follows from this experiment.
