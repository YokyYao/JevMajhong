# Fresh-base Mortal lineage: endgame retention

Status: prospective follow-up; preparation and fitting require the foundation's completed pipeline and successful final evidence audit. This does not modify the active foundation fit or publish a model. Configuration: [mortal-base-retention-v1.json](../../../configs/training/mortal-base-retention-v1.json).

## Why this phase exists

The fresh Qwen3.5-0.8B foundation learns22,321 fixed Mortal actions from East1–3 non-win menus. It has no direct East4 pursuit or win-acceptance supervision. The original task also retains endgame pursuit, so completion cannot be inferred from teacher agreement. Continue only the newly trained foundation's rank16 LoRA and head, after its validation selection, saved-loader checks and64-game development comparison finish. No old v8/128/176 adapter, head or optimizer is loaded. The frozen Qwen backbone and runtime architecture remain unchanged; no extra decision network is added.

## Data and target provenance

Use all48 previously selected native endgame TRAIN positions and8 separate monitor positions, with434 legal actions and64 paired terminal outcomes per action. Source membership was chosen without those outcomes. Hashes, world designs, completion/ledger receipts and terminal ranks were rechecked; these are not new simulations or independent han/fu scoring. Their source games are separate from the512 Mortal source games and the8 full-match comparison seeds.

Recompute their soft targets using the **new foundation's** normal-serving action distribution and the stored binary final-top-two outcomes. The existing paired-advantage recipe blends half raw and half uncertainty-shrunk advantage, with temperature0.1 and10% exploration. Keep all48 menus, including uncertain ones. Do not copy `prior128`, existing soft targets or labels. Fourth-place avoidance is reported, not a hard non-decrease filter; pursuit may take risks for top two. This target is terminal-outcome supervision, **not a Mortal teacher choice**.

Replay each source position and capture the actual `FourthChaseAgent` request before computing that prior or admitting training text. An input audit found5 held-position records with extra `screen_observation` absent from real serving; the other51 matched exactly. Use the captured native format, preserving the core public state and exact ordered candidate features. A dummy uniform response is used only to capture the request and is never applied to a source trajectory or used as a label. Reproduce all56 actual requests in post-fit auditing. Queued preparation/fitting/audit jobs from the first source revision were canceled before starting; no data or optimizer updates from that revision were admitted.

The reused outcomes still depend on old formal-v8/128 continuation policies and rule opponents. Current opponent concealed hands are fixed at their original native state while walls/future hands vary; they are not draws from a public-state posterior. Replacing the explicit prior does not remove these dependencies or make the labels expert truth. Native complete-game evaluation remains necessary. The small8-position monitor is an inspected development set.

Mix a reproducible2,048-example replay subset of the foundation's TRAIN teacher decisions with these48 outcome positions. Replay quotas: ordinary1,024; defense256; riichi-selected64/declined64; chi/pon-selected128/declined384; kan128. Select by identity hash, scarce strata first, at most16 decisions per source game. The foundation already trained on the full22,321 decisions; this smaller set is regularization for this phase, not a replacement claim about foundation coverage. Keep the same512 teacher validation decisions plus8 endgame monitor positions. Entire source games and decision IDs stay disjoint from training.

## Loss and schedule

A48-example accumulated update uses16 ordinary,8 defense,2 selected riichi,2 declined riichi,4 selected calls,6 declined calls,2 kan and8 terminal-outcome examples. Teacher families retain85% of the loss with the foundation's population-calibrated class mass; terminal outcomes receive15%. Balanced sampling does not impose a fictitious50/50 call prior. Teacher labels retain their existing2% smoothing; terminal targets retain their derived soft distribution.

Use a new AdamW optimizer: LoRA LR2e-6, head LR1e-5,16-update warmup, cosine decay to25%, weight decay0.001 and gradient cap1. Observe at least128 new updates and expose all2,096 admitted examples before selection/stopping. Save and validate every32 updates via the normal serving loader, BF16/unmerged, temperature1 and no prefix cache. Select the lowest joint validation CE (85% population-weighted teacher CE +15% monitor soft-target CE) after the observation floor. Stop after four stale checks with a0.1% relative improvement threshold, or384 updates. These are research selection rules, not playing-strength guarantees. Step0 must reproduce the audited foundation's teacher metrics exactly; both LoRA and head must subsequently change. Save optimizer, RNG and exposure state.

Report per-action teacher matches, ordinary/defense behavior, terminal-target CE and cached-outcome top-two/holding/fourth rates separately. A lower joint loss or a favorable8-position cached estimate cannot establish better play. Do not claim an endgame improvement solely from those labels.

## Complete-game comparison and completion

Reload selected and final checkpoints and reproduce their saved metrics. Run64 new complete East matches for the selected retention candidate using the same eight seeds, four seats, two opponent profiles, normal formatter and serving guard. Reuse both the audited176 reference games and the fresh foundation's completed64 candidate games; do not rerun either policy. Audit the same new candidate games against both references. Report first/top-two/fourth, mean rank/net points, hand wins/deal-ins and whole-seed paired uncertainty. This reused development grid is not independent confirmation. A positive confidence bound is not required merely to run the experiment; no statistical gain is promised.

Before declaring the task complete, verify exact fresh-foundation initialization, no old policy tensors/targets, complete source isolation/exposure, real optimizer updates, immutable registered sources/data, normal saved-model behavior, complete-game accounting and actual job exit/cleanup. A failed improvement remains a documented research result, not a reason to publish weights or an EXE. Only code, this plan and aggregate results are public. Raw records, detailed predictions and research weights remain private; no automatic main merge or release.

## Attribution

Decision interface: [JevAny](https://github.com/SimpleJev/JevAny). Fixed teacher: [Mortal](https://github.com/Equim-chan/Mortal), AGPL-3.0, with [VoidShine/mortal-298k weights](https://huggingface.co/VoidShine/mortal-298k). Pinned versions, native bridge and relevant notices remain in the [teacher plan](mortal-teacher-20261119.md) and the [fresh-base foundation plan](mortal-base-training-20261120.md). No upstream weights are redistributed by this experiment.
