# Targeted refresh: labeling complete, training resumed

The user authorized new targeted positions, better finite labels, a revised
mix and training from the128-step model. The plan and executable pipeline are
in `docs/experiments/plans/endgame-targeted-refresh-20261119.md` and
`configs/training/endgame-targeted-refresh-v1.json`.

Fresh collection completed2048 native games:1536 TRAIN source groups and512
separate MONITOR groups. Selection used public score deficits, dealer/threat
state and legal action type before counterfactual outcomes. It froze48 TRAIN
menus and8 MONITOR menus, all from distinct new source games. The old186
validation rows are preserved unchanged.

| Kind | TRAIN | MONITOR |
|---|---:|---:|
|Fourth discard/riichi|16|2|
|Fourth chi|4|1|
|Fourth pon|4|1|
|Fourth kan|4|1|
|Fourth small-win/continue|4|1|
|Third pursuit|8|1|
|Leading holding|8|1|

The training kan quota includes actual legal ankan, daiminkan and kakan
positions. All legal actions are compared:434 actions total, balanced218/216
between workers, each action with64 paired worlds,27,776 complete rollouts.
Current hidden opponent hands remain fixed to the native source; the two
replication blocks vary unseen walls/future deals and balance frozen hero
policies and opponent styles. These are conditional simulation estimates,
not expert truth or a full hidden-hand posterior.

Collection GPU309281 and both label workers309282/309283 exited0. All56
menus and27,776 complete branch rollouts are labeled. Initial verification309284
and regression309295 stopped at GPU preflight because the allocated GPU already
had other compute processes; dependent training309315 auto-canceled. Those
processes were not touched and the completed labels were retained.

Recovery assembly309621 exited0 and verified874 positive training rows,8 fresh
monitoring menus and the unchanged186 old validation rows. Combined GPU job309626
on ws-l2-012 passed277 core/89 desktop checks (one optional skip), then started
training in the same clean allocation. Every stage remains bounded and writes
logs and exit receipts. Training has passed total step144 (16 new updates).

The first saved-and-normally-loaded validation is below. Both columns use the
same respective held-out menus and finite simulation labels. The combined
metric equally weights the old and fresh cohort means; it is not a full-game
or human-match top-two rate.

| Fourth-place validation cohort | Parent128 | Step144 | Change |
|---|---:|---:|---:|
|Old51 menus|22.55%|20.47%|-2.08 pp|
|Fresh6 menus|23.70%|24.48%|+0.78 pp|
|Combined primary|23.12%|22.47%|-0.65 pp|

There is no overall improvement yet and no selected replacement checkpoint.
Training continues under the registered observation/patience schedule. The
23.12% combined baseline is the same parent128 weights on a different cohort
mix; it must not be presented as a gain over the old51-only22.55% baseline.

Data source94a74d33/CI44534005698 and GPU309279 passed273 core/89 desktop
checks (one optional skip). Training sourcecd5f8db5/CI44536025274 passed277
core/desktop workflow checks; earlier GPU309288 passed276 core before the
additional metadata test. Final277 core checks passed in recovery309626.
Initial unsupported live-seed test309263 failed before data
generation. The corrected native-boundary reset now passes next-deal variety
and full-match ledger tests; probe jobs309265/267/269/273 are terminal.

The verified positive training set is874 rows (existing826 plus48 fresh),
with40% fresh loss mass,45% ordinary/leading replay,10% old fourth targets
and5% old third/holding. Start exactly from128 LoRA/head, with a new optimizer
and warmup for the changed objective, not from384. Normal loaded checkpoints
are evaluated every16 new updates, maximum192, observation64 then patience3.
Fresh8 monitoring remains outside gradients. A later complete-continuation
comparison is required before any promotion; no independent/full-start pool,
formal model/main/EXE or public research weights changed.

Only code, plans and aggregate findings are public. Full source games, paired
records, features and research checkpoint files remain private.
