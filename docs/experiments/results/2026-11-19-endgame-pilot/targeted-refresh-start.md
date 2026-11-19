# Targeted refresh: source coverage complete, paired labeling running

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

Collection GPU309281 exited0. Label workers309282 and309283 are running on
ws-l1-001/ws-l1-006. Data verification309284 is afterok both. New fitting and
saved-weight audit309315 are afterok verification AND final regression309295.
Invalid dependencies auto-cancel; each stage has bounds, logs and exit receipts.
Cluster QoS currently admits two running user jobs, so the final cluster
regression waits while both label GPUs work. It must pass before fitting.
No new model parameter update has been claimed at this labeling stage.

Data source94a74d33/CI44534005698 and GPU309279 passed273 core/89 desktop
checks (one optional skip). Training sourcecd5f8db5/CI44536025274 passed277
core/desktop workflow checks; earlier GPU309288 passed276 core before the
additional metadata test. Final GPU309295 is a dependency, not reported as
already passed. Initial unsupported live-seed test309263 failed before data
generation. The corrected native-boundary reset now passes next-deal variety
and full-match ledger tests; probe jobs309265/267/269/273 are terminal.

The planned positive training set is874 rows (existing826 plus48 fresh),
with40% fresh loss mass,45% ordinary/leading replay,10% old fourth targets
and5% old third/holding. Start exactly from128 LoRA/head, with a new optimizer
and warmup for the changed objective, not from384. Normal loaded checkpoints
are evaluated every16 new updates, maximum192, observation64 then patience3.
Fresh8 monitoring remains outside gradients. A later complete-continuation
comparison is required before any promotion; no independent/full-start pool,
formal model/main/EXE or public research weights changed.

Only code, plans and aggregate findings are public. Full source games, paired
records, features and research checkpoint files remain private.
