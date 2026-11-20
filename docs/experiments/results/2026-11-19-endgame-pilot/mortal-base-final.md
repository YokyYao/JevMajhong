# Mortal teacher: fresh-base training results

The latest requested route starts from pinned Qwen3.5-0.8B with a new rank16 LoRA and action head, using local VoidShine/Mortal-298k as teacher. It does not initialize from v8,128 or176. The pretrained backbone stays frozen; “from base” means a fresh Mahjong adaptation, not training every backbone parameter from random weights.

The teacher foundation and separate fresh-lineage terminal-retention run are complete. The retention candidate recovered much of the foundation's playing deficit on the reused development grid, but did not demonstrate stronger play than the audited176 reference. No model replacement or EXE release is justified by these results.

## Actual lineage and data

1. **Fresh foundation:**22,321 teacher TRAIN decisions from384 complete source games;512 validation menus selected from128 separate source games. New LoRA/head, frozen base, new optimizer. Teacher family loss mass45% ordinary discard,15% defense,15% riichi,20% calls and5% kan; accept/decline weights reflect source frequencies. Minimum768 updates and full data exposure; validate saved normal-loader checkpoints every64. Training stopped1408 after four stale checks; selected1152.
2. **Fresh-lineage terminal retention:** initialize only from that fresh1152 adapter/head, reset optimizer. Train2,048 teacher replay decisions plus48 terminal-outcome menus; validate512 teacher plus8 terminal monitor menus. Loss85% teacher/15% terminal. Targets use fresh1152 priors and existing paired64 final-top-two outcomes, not old priors or copied old soft targets. Minimum128 new updates/full exposure, checks every32, maximum384. Training stopped new256 after four stale checks; selected new128, checkpoint index1280 in a separate directory.

All admitted data were exposed; selected/final saved metrics reproduce on normal reload. Step numbers are experiment-local: retention checkpoint1408 is not foundation checkpoint1408. The foundation's discarded updates after1152 are not the retention initialization.

Terminal outcomes still depend on old v8/128 continuation policies and fixed original concealed hands. Reusing those outcomes is disclosed data reuse, not initialization from those policies, Mortal expert targets, or sampling from a public-state posterior. Full source-game splits are enforced. The tiny8-position terminal monitor is not an overall win-rate test.

## Complete East-game development comparison

Each policy has64 complete East matches:8 shared seeds,4 controlled seats and2 opponent profiles. Each profile has two Efficiency opponents plus one Calling or Mortal opponent. The controlled model receives public-state requests. The64 new retention games reuse the already audited176 reference and fresh1152 foundation games; neither baseline policy was run again.

| Policy | Top-two | First | Fourth | Mean placement | Net points/game | Hand-win rate |
|---|---:|---:|---:|---:|---:|---:|
| Audited176 reference |31/64,48.44%|17.19%|25.00%|2.59375|−1,445.31|20.63%|
| Fresh teacher foundation,1152 |24/64,37.50%|15.63%|25.00%|2.71875|−2,350.00|15.32%|
| Fresh-lineage retention,new128 |30/64,46.88%|23.44%|23.44%|2.53125|−1,634.38|18.66%|

Retention versus fresh1152: **+9.375 percentage points** top-two, whole-seed95% t interval **[−1.44,+20.19]pp**; **+715.625 points/game**.

Retention versus audited176: **−1.5625pp** top-two, interval **[−11.92,+8.79]pp**; **−189.0625 points/game**. Its higher first rate and better mean placement do not override the top-two/net-point comparison.

These are reused development seeds inspected during research, not independent confirmation or online Mahjong Soul performance. There are8 seed clusters, not64 independent paired seeds; both intervals cross zero. Do not present the recovery over the weak fresh foundation as proof of an improvement over the old reference.

## What the result teaches

Teacher validation loss is not playing strength: foundation1152 had the best eligible teacher CE but worse measured top-two/net points than its768 development screen. Retention joint CE also rose from0.74527 at its parent to0.75433 at selectednew128, despite better complete-game point estimates than that parent. Future selection should keep teacher CE as a diagnostic and use a separately registered, small match screen for playing decisions, followed by new held-out seed groups.

Call/kan supervision still needs work. Fresh foundation / retention / reference actual64-game actions were respectively: pon33/42/89, chi10/11/60, ankan7/5/9, kakan0/0/4 and daiminkan0/0/21. Counts depend on encountered opportunities and are not acceptance rates or evidence every reference call was optimal. Existing teacher TRAIN contains20 positive ankan and10 positive kakan examples, but **no positive daiminkan**. A four-window bridge check confirmed that daiminkan was legal in the teacher mask and correctly decoded, while the natural teacher choice preferred alternatives; it did not establish a universal teacher aversion. Collect targeted legal windows and preserve the teacher's natural recommendations, instead of inventing positive kan labels.

Terminal corrective pressure was sparse: only3/48 admitted targets changed the fresh parent's greedy action;1/8 monitor targets did so. Raw simulated best-action gaps are not expert truth. A next revision should add diverse current-policy terminal examples and more reliable public-state outcomes, with uncertainty treatment that preserves useful action distinctions. Change the recipe between registered experiments, not midway through this one.

## Evidence, scope and publication

Foundation fit309978 and evidence audit309982, retention preparation310254, fit/evaluation310255 and evidence audit310256 all finished with controller state COMPLETED and exit0:0. Retention audit independently verified exact fresh-parent tensors, all376 optimizer states at selected/final update counts, full2,096-example coverage, all56 actual native requests and regenerated targets using the fresh prior, normal saved metrics, source/runtime identities and both complete-game comparisons. The release-model identity remains unchanged.

Each comparison reconciles128 complete-game payment/pool/rank records. Shared games across comparisons are reused, not additional matches, and this accounting check is not an independent han/fu scorer. Source a037dd44 passed core/desktop CI44650768159; the five retention checks passed. This final report changes documentation only.

Public artifacts include source, registered plans and aggregate reports in [PR7](https://github.com/YokyYao/JevMajhong/pull/7). Raw hands/events, individual predictions and research weights stay private. The released v8 model is unchanged. No automatic main merge, model/EXE export or additional training run follows from these development results.

Details: [foundation trajectory](mortal-base-progress.md), [retention trajectory](mortal-base-retention-progress.md), [foundation plan](../../plans/mortal-base-training-20261120.md), [retention plan](../../plans/mortal-base-retention-20261120.md).
