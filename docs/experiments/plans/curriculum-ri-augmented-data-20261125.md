# Prospective augmented RI data and complete-coverage replay

Keep the ongoing2472-update trial frozen. Prepare a separate nineteen-source union
of its exact87595 TRAIN/23688 VAL rows and the independently audited256-family
basic teacher supplement(4975 TRAIN/1310 VAL). This yields92570 TRAIN/24998 VAL.
The original bytes remain an exact file prefix; new normalized teacher requests
change only their private loss-source tag. Preserve original states, labels,
masked targets, IDs, family splits and all ancestor exclusions.

Keep all19 registered source masses. Only the three basic teacher class counts
grow:plain3106 ->6190, RI accepts1075 ->2123, declines812 ->1655. Legacy RI remains
194 accepts/272 declines. Total RI TRAIN is4244. New data adds independent cases;
it does not increase the teacher origin's total objective mass. Recompute bounded
per-original weights against the entire new TRAIN population. Keep original VAL
and new VAL IDs as separate cohorts so improved aggregate metrics cannot hide
retention losses on the previously declared source-quality checks.

The old sampler takes exactly one member of each RI source per update. It fails
full coverage when a source has more originals than updates. Do not truncate
the2123 teacher accept examples, raise the legacy repeat cap or claim untrained
examples were consumed. The new sampler keeps at least one member of every RI
source per update, adds all remaining RI originals across updates, and preserves
the original non-RI subsequence with only bounded padding.

Choose the smallest full-batch update count U for which
`N + sum(max(0, U - RI_source_count)) <= 73 * U`.
Distribute excess RI originals evenly across those updates. Inverse-repeat
weights preserve each original's contribution `source_mass/source_count` over
an epoch. Retain repeat cap8 and individual weight cap16. This preserves the
empirical fixed-parameter objective, not AdamW/dropout trajectories. Any failure
of coverage or caps must reject the schedule rather than silently drop data.

Freeze two epochs using the declared baseline/exposure seeds in the JSON recipe.
Derive the complete update and exposure budget from actual counts, then retain
the actual two epoch plans and their hash bindings. No VAL oversampling or source
repartition is permitted. The generic existing epoch adapter must accept both
plans before any full-model integration.

For the prospective fixed-strength RI-parent KL objective, reuse the already
bound2353 qualified-parent TRAIN predictions only after proving exact original
input membership and model identity. Infer the1891 new RI TRAIN menus with that
same qualified calls388 parent. Bind all4244 target menus to the new full TRAIN
file before optimizer updates. Do not infer VAL to manufacture training targets.
Mortal supplies new hard action labels; the earlier student supplies the separate
retention probabilities. Neither teacher runs during deployment.

Full combined token admission, actual weighted observer integration, full-model
optimizer/save/reload control and fit registration remain mandatory. A future
fit may start only after the ongoing trial completes and fails its original
strategy criteria with a successful execution audit. A pass follows its existing
3B path. Data preparation or an interim negative result cannot start this fit.
Keep full-budget selection, declared loss/reward/win/top-two/net observations,
original retention and native requirements, and the unopened reserved comparison.
Only code/recipes/aggregate counts are public; do not publish raw targets or
automatically replace v8, merge main or release an executable.
