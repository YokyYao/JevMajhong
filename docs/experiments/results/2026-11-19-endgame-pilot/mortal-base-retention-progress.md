# Fresh-base lineage: endgame-retention progress

Status: optimizer updates stopped at new256 after four stale checks beyond the128-update/full-exposure floor. All2,096 admitted examples were exposed. New128 is the selected research checkpoint; its joint loss remains above the parent baseline. GPU job310255 is now completing saved-checkpoint reproduction and the64-game comparison; dependent audit310256 remains required. The [registered plan](../../plans/mortal-base-retention-20261120.md) was unchanged. Selection is not an improvement or release qualification.

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
| 64 | 0.75655 | 0.70352 | 0.74859 | 2,096/2,096 | Observation floor not met |
| 96 | 0.76371 | 0.71483 | 0.75638 | 2,096/2,096 | Observation floor not met |
| 128 | 0.76332 | 0.70337 | 0.75433 | 2,096/2,096 | First eligible research candidate; not better than parent |
| 160 | 0.77559 | 0.72554 | 0.76808 | 2,096/2,096 | Retain128; one stale check |
| 192 | 0.76645 | 0.71620 | 0.75891 | 2,096/2,096 | Retain128; two stale checks |
| 224 | 0.78232 | 0.73460 | 0.77516 | 2,096/2,096 | Retain128; three stale checks |
| 256 | 0.78031 | 0.72485 | 0.77199 | 2,096/2,096 | Retain128; four stale checks, stop updates |

At32, both LoRA and head have changed, but joint loss is slightly above the baseline. Exact teacher-positive/negative call matches move40/64 and63/64 →39/64 and63/64. Riichi has27 teacher-positive action-type matches with17 extras. No positive daiminkan teacher supervision has appeared. These are auxiliary teacher matches, not independently correct action labels or win rates.

At64, both parameter groups still change and every admitted example has been exposed, but the128-update floor is not reached. Teacher CE is almost identical to the parent, terminal-target CE is higher, and joint CE0.74859 remains above the0.74527 baseline. Exact positive/negative calls are40/64 and63/64; riichi has27 positive action-type matches with18 extras. No benefit is established by the minor teacher-loss fluctuation.

At96/128, joint CE remains above the baseline. New128 satisfies the observation/exposure floor and becomes the first eligible research checkpoint; this rule does not claim it improves on step0. Exact teacher-positive/negative call matches at128 are38/64 and63/64. Continue the registered post-floor checks, and require eventual complete-game comparison rather than treating selection as a quality pass.

At160, both teacher and terminal-target CE worsen, giving joint CE0.76808 and the first stale post-floor check. Keep128 as the research candidate. The cached hard estimates below remain the same as96/128, so no improvement is established. Continue only within the registered four-stale-check/384-update bounds.

## Cached terminal monitor, not complete-match top-two rate

| Initial placement stratum | Monitor positions | Parent new0 cached top-two estimate | New32 | New64 | New96 | New128 |
|---|---:|---:|---:|---:|---:|---:|
| Fourth | 6 | 24.48% | 24.48% | 24.48% | 23.70% | 23.70% |
| Third | 1 | 46.88% | 46.88% | 46.88% | 46.88% | 46.88% |
| First/second holding | 1 | 43.75% | 43.75% | 43.75% | 43.75% | 43.75% |

New160/192/224/256 have the same cached hard estimates as128 in all three strata. None of the saved checkpoints beats the parent joint-loss baseline. The registered run stops further optimizer updates rather than changing labels or selection rules mid-run.

These values apply the model's chosen action to stored counterfactual outcomes, with old continuation behavior after that action. They are not freshly played games, an independent test set or the model's overall top-two rate. New32/64 are unchanged and96/128 are slightly lower in the fourth-place stratum; the tiny8-position monitor cannot establish general playing strength. No benefit is claimed from small soft-distribution changes. The observation floor is now met; post-floor fitting remains bounded by the registered stopping rule.

## Corrective-label signal versus greedy serving

Because the foundation's match result was negative and the terminal monitor remained flat, a bounded read-only diagnostic inspected the admitted targets and existing saved predictions. It loaded no policy weights, generated no new games and changed no labels, parameters or recipe. Its GPU-allocation step exited0 and was reaped.

Compare each target's maximum-probability action with the fresh parent's recorded greedy action:

| Mutually exclusive TRAIN source stratum | Positions | Target greedy action differs from parent |
|---|---:|---:|
| Win menus | 4 | 2 |
| Discard stratum | 16 | 1 |
| Chi/pon/kan strata | 12 | 0 |
| Third-place stratum | 8 | 0 |
| Holding stratum | 8 | 0 |
| Total | 48 | 3 |

These are source strata, not counts of every available call/kan opportunity; a menu can contain several action types. All3 changed targets have positive **cached** mean advantage over the parent's action. Among8 monitor positions, only1 target changes the parent's greedy action;7 already share that action. Saved0/32/64 predictions match7/8 target greedy choices, and96 matches6/8 after one other action changes unfavorably in the cached outcomes.

There are29 TRAIN positions with a raw empirically higher-valued alternative, but26 keep the parent's target argmax after prior tilt and uncertainty treatment. Those raw maxima are optimistic over reused64 outcomes under old continuation policies; they are not29 expert-confirmed mistakes. The diagnostic identifies sparse direct pressure to change the deterministic serving action. It does not prove that soft training cannot generalize, that all unchanged labels are wrong, or that simply forcing raw maxima would improve play. The active protocol remains frozen; this evidence belongs in the final method analysis, rather than motivating an unrecorded mid-run label change.

## Remaining evaluation

After fitting, reload selected/final checkpoints and reproduce their metrics, then run64 new complete East games for the selected retention candidate. Reuse both the audited176 reference games and the fresh1152 foundation games; rerun neither baseline policy. Compare against both with whole-seed uncertainty and independent payment/pool/rank reconciliation. Dependency-bound GPU audit310256 also checks real optimizer states/exposure, exact parent lineage, all56 normalized inputs/targets and source/runtime identities. These reused development comparisons are not independent confirmation.

Optimizer fitting is complete; complete-game evaluation and final evidence remain incomplete. Raw records, detailed predictions and weights stay private; no automatic model/EXE release or main merge follows from this experiment.
