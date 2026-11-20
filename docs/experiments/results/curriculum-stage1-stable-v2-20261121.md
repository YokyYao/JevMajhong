# Stable-v2 stage1: interim results and riichi diagnostics

2026-11-21, Asia/Dubai. GPU training311884 is active with a fixed four-epoch/584-update budget; dependent audit311887 has not started. This is an interim development report, not a completed model or evidence of v8-level strength. The registered final584 checkpoint remains the sole candidate.

## First full epoch

The recipe preserves the first run's9,292/1,972 source-isolated decisions, rank16 LoRA/256-dimensional pointer, seed, source/base identity, loss masses and peak learning rates. It starts fresh from base. Changes are without-replacement64-record accumulation groups containing both riichi labels and four epochs. That is a combined recipe/exposure experiment, not a causal batching ablation.

At the203-update health observation, all update records are finite and contiguous, actual category counts match the epoch scheduler, source-weighted losses reconcile, and all registered inputs remain unchanged. The first146 updates contain exactly8,826 plain +194 riichi-selected +272 riichi-declined =9,292 records. The saved146 optimizer has376 parameter states all at146, epoch1, next-microbatch0 and records-seen9,292. Initial4 head and372 adapter tensors match the first run exactly, and actual LoRA/head updates were verified after8 updates. These checks prove training execution, not decision quality.

| Update | Weighted validation CE | Plain exact agreement | Teacher-riichi accepted | Teacher-decline correctly declined |
|---:|---:|---:|---:|---:|
|0|2.473897|15.07%|3/37|69/70|
|73|1.891910|30.99%|5/37|66/70|
|146|1.546582|44.13%|18/37|24/70|
|219|1.438572|47.77%|20/37|34/70|

CE/plain imitation improve, while riichi changes from conservative at73 to over-accepting at146. At219 both acceptance and correct decline improve relative to146, but remain54.05%/48.57%, below the two-sided phase target. Keep the fixed recipe through its registered budget; do not silently select an early point or change inference from these diagnostics. Update219 has no new game evaluation; the next saved-checkpoint development games are at292.

The matching16-game development grid at146 gives hand-win16.04%, dealin18.30%, first12.50%, top-two43.75%, fourth25.00% and net−3,718.75 points/match. The initial grid is top-two6.25%/net−11,437.50. Frozen v8 on the same16-game subset gives top-two62.50%/net+3,612.50. Only two seed groups underlie these points; they are not independent confirmation or a precise population-strength estimate. V8's cached reference is reused without new games.

## Discrimination is weak, not just a convenient threshold problem

The saved-score audit uses the exact same107 already-consumed validation riichi windows for both curriculum runs. It loads no model, plays no games and changes no policy. It distinguishes the current flat candidate argmax from a diagnostic comparison of reach probability mass against total discard probability mass. It also examines the reach-versus-best-discard logit margin, which represents a possible constant reach-bias adjustment.

| Run / point | Reach-mass AUC | Flat-margin AUC | Best balanced accuracy from a margin threshold on consumed validation | Could that threshold give both recalls≥60%? |
|---|---:|---:|---:|---|
| Stable-v2,0 |0.5436|0.5703|0.5853|No|
| Stable-v2,73 |0.6143|0.5826|0.6154|No|
| Stable-v2,146 |0.4645|0.4305|0.5515|No|
| Stable-v2,219 |0.6035|0.5699|0.6114|No|
| Initial curriculum v1, registered final1162 |0.5328|0.5004|0.5647|No|

These are small development-set descriptions without population-confidence claims. An oracle threshold is an upper-bound diagnostic on inspected labels, not an available deployment threshold. The script does not return or deploy such a threshold. Average balanced accuracy above60% does not imply that each recall passes60%. Ranking at219 improves after the146 decline but still varies across updates; this does not establish stable discrimination. Earlier transient checkpoints can look different; they do not replace either registered final candidate.

At146,10 of46 false-positive reach selections occur even though total discard probability mass is larger. Grouping types at0.5 reduces false positives to36 but also reduces true positives from18 to15, leaving balanced accuracy44.56%. Thus this particular aggregation does not solve the problem. A fixed bias on the current reach/best-discard margin cannot meet the declared two-sided development target either.

All107 reach candidate objects contain only the `type` field, while discard candidates expose shanten, ukeire and safety features. The full player-visible hand/board state is still present; information is not necessarily absent from the complete request. This suggests testing a more symmetric, public-derived feature representation if the completed run remains weak, but it does not prove that feature asymmetry caused the bias. Any future feature change must be explicit, label-independent, identically applied in training/inference, separately frozen and compared against the practical v8 path. Do not change the active representation mid-fit.

The data checks also find zero already-declared-self-riichi windows and no missing selected-discard shanten in the70 teacher-decline cases. Sixty-two teacher declines keep tenpai and eight break it, consistent with the need to learn value/risk rather than always reach whenever available. Observed expert choices remain imitation targets, not optimal-action certificates.

## Explicit parent selection for the next course

The new `tools/curriculum_parent.py` admission path names the audited registered final checkpoint, not the legacy research result's minimum-CE summary. It requires completed fit/audit exits, actual two-sided riichi/basic retention metrics, unchanged inputs and source packages, matching saved-model identity, and held-out/reserved source-group exclusion in the proposed child. Stage2 must retain the original stage1 replay. A seal binds model/data/code/audit artifacts without starting a child fit or resuming an old optimizer.

Five gate tests pass, including rejection of a misleading passed flag, a partial budget, a minimum-CE replacement and a missing quality gate. Real negative checks reject both the still-incomplete stable-v2 fit and the completed original curriculum without a passed riichi gate, writing no parent seal. The completed positive end-to-end sealing/warm-start path cannot be claimed until a real parent passes its final audit and stage target. No stage2 fit is started by these checks.

All21 related curriculum checks passed on allocated GPU companion resources. New raw predictions, gradient details, native game events and research weights remain private. The [four-stage goal](../plans/curriculum-optimization-20261121.md), later-course fitting and unseen final comparison remain outstanding.
