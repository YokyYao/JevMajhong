# Curriculum stage1 reproduction

This is the newly registered basic-discard/riichi course, not a replay of every historical v8 experiment. The student initializes a fresh rank16 LoRA and256-dimensional action head on pinned Qwen3.5-0.8B. No v7/v8/Mortal weights initialize it. The frozen v8 export is used only by a separate evaluation process.

The first registered fit completed1,162 updates and its final64-match development comparison. See the [completed stage1 report](../docs/experiments/results/curriculum-stage1-20261120.md) for all validation points, the unresolved riichi bias and the comparison with v8. A successful training exit is not a release qualification.

## Environment and public inputs

Run all commands below in a scheduled GPU allocation. On this workspace, running them on login, including data processing/tests/plotting, is prohibited. The observed package versions are in `curriculum-runtime.txt`; PyTorch needs a CUDA build compatible with the host. Use the matching JevAny package and source restoration below rather than an unpinned latest install.

Required public inputs:

- The dataset at fixed repository commit `7ee572904bfd30387d2451a2dd1d770f90ffebf7`, recovered automatically by `prepare_curriculum.py` with every archive/file hash checked.
- Qwen/Qwen3.5-0.8B at revision `2fc06364715b967f1860aea9cf38778875588b17`. Download that exact snapshot into `BASE_SNAPSHOT`; `curriculum-base-files.json` checks its ten files. A model download is not a training step.
- The public Jev Play v2 export: restore the repository's checkpoint parts using `tools/restore_weights.py`, setting its destination to `models/jev-play-v2`. No historical v7 parent or optimizer is required. The registry checks the export files and v8 head tensors; its metadata-only export has a different file-bundle hash from the original training checkpoint while preserving policy parameters.

The source bootstrap retrieves only the enumerated code Git blobs at fixed public commits, then applies the two published overlays. It verifies the exact training-package digest `7a6e9471c95d529ecdc5d0a66cd240ae17d1f5914d76ce8dbd61ad45ce7104bc` and JevAny digest `84fe7f0201a8d9928aabe4770912bf25e76c4916c1eded3011cf8f02862b758d`. The initialization overlay keeps the actual research package identical; fresh stage1 does not call its historical-parent loading path. The metric journal is the second overlay. Upstream code retains its original licenses, notices and attribution.

## Commands inside the allocation

Use a fresh output directory; these tools reject overwriting prior runs. The exact configuration below was registered for9,292 training decisions,1,972 validation decisions, two epochs and1,162 actual optimizer updates.

```bash
export JEV_BASE_MODEL_DIR="$BASE_SNAPSHOT"
python tools/bootstrap_curriculum.py
python tools/prepare_curriculum.py --out runs/curriculum-reproduction-v1/public-data
python tools/build_curriculum_stage1.py \
  --source runs/curriculum-reproduction-v1/public-data \
  --out runs/curriculum-reproduction-v1/clean-data-v1
python tools/register_curriculum.py \
  --config configs/training/curriculum-stage1-v1.json \
  --base-path "$BASE_SNAPSHOT" \
  --reference-path models/jev-play-v2
python tools/train_curriculum.py \
  --registered runs/curriculum-reproduction-v1/fit-plan/registered.json
```

`cluster/curriculum-stage1.slurm` provides a bounded foreground launch with logs and an exit receipt. Create the log directory and set the base-snapshot environment variable before submitting it; change the partition for another cluster. Prepare/install the recorded Python environment inside an allocated GPU job too. The script does not overwrite existing data or restart an existing fit.

The data builder reserves20% of public TRAIN source groups with seed2026112001 before cleaning and removes these groups from all later-stage training. It never uses reward or labels to choose held-out groups. Expert/MC actions are unchanged; source-only loss tags do not enter model input. Unknown dealer/wind/score/riichi/hand-number context is isolated rather than invented. See the [coverage and shape audit](../docs/experiments/results/curriculum-data-shape-audit-20261120.md).

This new split is valid for the fresh student. V8 previously trained on some of these newly held-out source groups, so these validation menus must not be presented as an unseen supervised test for v8 or a student initialized from v8. The actual policy comparison uses the separately registered complete-game seed grid.

The registry writes the resolved trainer configuration, dataset and source hashes. Machine-dependent file paths and metadata may change file-bundle hashes across hosts. Compare the registered numerical configuration, data hashes, model tensors and metric trajectories; bitwise weight-file identity across different CUDA/runtime/hardware is not promised.

## Real logs and curves

`stage1-fit/metrics/train_metrics.jsonl` records every real optimizer update, source losses/counts, actual upstream loss, gradient norm and learning rate. Supervised reward is null. `validation_metrics.jsonl` records the fixed-stratum and natural validation losses, riichi-selected/declined counts and accuracy. `evaluation_metrics.jsonl` records actual complete East matches, terminal net-points/8000 reward and separate hand-win/first/top-two/fourth rates. The model isn't trained by these diagnostic games.

Step0 records the fresh student's pre-training validation and16 complete games. V8 has64 reference games on the same rule/opponent/seat grid. Every256 updates uses the same16-game grid; the final checkpoint has64 games, including that grid as a cached subset. Its source seeds are development data after inspection, not independent final-test seeds. The final-two-epoch checkpoint is preregistered; neither minimum CE nor the highest noisy16-game rate silently replaces it.

After successful fitting, raw CSV and standalone PNG/PDF figures are generated. They can be regenerated without training:

```bash
python tools/plot_curriculum_fixed_grid.py \
  --metrics-dir runs/curriculum-reproduction-v1/stage1-fit/metrics
```

Use this aligned plot for comparisons: all curve points use the same16 development matches, including the matching subset of the final64. The full final64 statistics stay in the original logs and `evaluation-stage-end64.csv`. `plot-data-provenance.json` records the transformation; no new games are generated. The raw training wrapper's original plot is replaced by this matched-grid rendering when running the cluster template or the supervised end audit.

The initial model is a random Mahjong head/adapter on a pretrained backbone, not a random-action agent; the legal-action/efficiency guard and forced legal wins still apply. Do not call its initial game results a learned Mahjong ability. Stage1 has no win/pass or yaku-scoring supervision. Stage2/3 data are prepared with the same holdout exclusions; their fits and stage4's public data generation are separate registered work and are not automatically launched by this command.

Raw match events and new weights stay local. Only aggregate results/curves are intended for publication. Existing v8 parameters and the Windows tool are unchanged.

## Registered stable-v2 follow-up

The [optimization plan](../docs/experiments/plans/curriculum-optimization-20261121.md) keeps the full four-course goal and records why the first fit is not a v8 replacement. The next fresh-base fit uses the same clean data/loss weights with without-replacement64-record accumulation groups containing both riichi labels, four epochs and584 updates. This changes batch construction and data exposure together; it is not a single-factor ablation. It does not continue the original stage1 or v8 adapter.

After the public source/bootstrap and clean data above are prepared, register a fresh fit inside the GPU allocation:

```bash
python -m unittest discover -s tests/core -p test_curriculum_order.py -v
python tools/register_curriculum_stable.py \
  --config configs/training/curriculum-stage1-stable-v2.json \
  --base-path "$BASE_SNAPSHOT" \
  --reference-path models/jev-play-v2
python tools/train_curriculum_stable.py \
  --registered runs/curriculum-optimization-v2/fit-plan/registered.json
python tools/audit_curriculum_fit.py \
  --registered runs/curriculum-optimization-v2/fit-plan/registered.json \
  --out runs/curriculum-optimization-v2/final-evidence-private.json
```

`cluster/curriculum-stage1-stable.slurm` provides a bounded launch. The shipped config reuses the completed original experiment's v8 reference cache and checks its identity/grid/files. If that cache has not been generated, make a local copy of the config, set `reference_evaluation_cache` to null before registration, and pass that config instead; the trainer will compute the same reference once. This does not require any old student weight. Public v8 exports may have a metadata-only file-bundle identity difference; the registrar retains the pinned tensor verification.

The epoch scheduler only changes the pinned trainer's epoch-order RNG; per-item augmentation RNGs remain ordinary. Token admission must preserve the exact data-file IDs/order, each epoch must be a unique complete permutation, and actual optimizer logs must contain both riichi classes at every update. Receipts preserve per-update counts and order digests privately. The final audit reconciles these with observed training logs and optimizer state; its evidence-pass flag is distinct from the stage-quality gate and does not mean the full four-course goal is achieved.

## Later-course data preparation

[Stage2/3 preparation results](../docs/experiments/results/curriculum-stage2-stage3-data-20261121.md) document executable coverage audits/builders, source identities and all proposed loss masses. Stage2 retains20,237/4,721 expert decisions with unchanged visible inputs/labels. Stage3 structurally quarantines136/41 single-support masked targets, leaving22,959/5,905 informative decisions; preserve its full original validation as a separate legacy diagnostic. Neither preparation command starts a model fit or admits a transfer parent. Do not splice the changed validation population into an earlier comparable-loss curve. All preparation commands also require a scheduled GPU allocation.

The public `tools/curriculum_losses.py` module supports offline core regression checks without importing research files absent from the runtime tree. `tools/check_curriculum_losses.py` separately verifies values/gradients against the restored, digest-checked research package on real expert/simulation/quarantined records. The current trainer stays frozen. [Stage4 native win/decline checks](../docs/experiments/results/curriculum-stage4-interface-20261121.md) also remain separate from data admission and fitting.

The [completed stable-v2 report](../docs/experiments/results/curriculum-stage1-stable-v2-20261121.md) distinguishes actual update/sample integrity from unresolved riichi discrimination and v8-strength gaps. `tools/audit_curriculum_riichi_scores.py` does not deploy a validation-selected threshold. `tools/curriculum_parent.py` defaults to the original strict final-checkpoint admission; legacy minimum CE is not a transfer decision. A separate, explicitly bound research-only amendment now admits the completed final584 without relabeling its failed original riichi gate. Actual sealing, qualified-parent reading and exact adapter/head warm start have executed; [the calls plan](../docs/experiments/plans/curriculum-stage2-calls-20261121.md) records the status.

## Controlled context-v3 input experiment

The [context report](../docs/experiments/results/curriculum-riichi-context-20261121.md) describes the public feature transform, diagnostic probe and controlled fresh-base recipe. A probe is not loaded into Jev. The active stable-v2 files must remain unchanged. Inside scheduled GPU resources, after stable-v2 registration provides the original admitted data identities:

```bash
python tools/prepare_curriculum_riichi_context.py \
  --registered runs/curriculum-optimization-v2/fit-plan/registered.json \
  --out runs/curriculum-optimization-v2/riichi-context-data
python -m unittest discover -s tests/core -p 'test_curriculum*py'
python tools/register_curriculum_context.py \
  --config configs/training/curriculum-stage1-context-v3.json \
  --base-path "$BASE_SNAPSHOT" --reference-path models/jev-play-v2
python tools/train_curriculum_context.py \
  --registered runs/curriculum-context-v3/fit-plan/registered.json
python tools/audit_curriculum_context_fit.py \
  --registered runs/curriculum-context-v3/fit-plan/registered.json \
  --out runs/curriculum-context-v3/final-evidence-private.json
```

Use fresh output directories; preparation/registration reject overwrites. As with stable-v2, a missing original reference cache requires setting `reference_evaluation_cache` to null in a local config before registration. `cluster/curriculum-stage1-context.slurm` runs an already registered manifest on bounded GPU resources; queue its final audit only after a successful fit, cancelling invalid dependencies.

`tools/preflight_curriculum_context.py` combines registration and actual107-menu input parity on a private copy of a stable-v2 step0 checkpoint. It is an additional diagnostic requiring that checkpoint; the copy is never a training initializer. Use it instead of the standalone registration command, with `--config`, `--registered`, `--base`, `--reference` and `--out` paths. Our recorded preflight311989 completed on a GPU with0 maximum probability difference. All new model checkpoints bind their input schema/feature code/model files in a sidecar; later-course registration must retain the same input contract. The normal initialization remains fresh base with a fresh optimizer and no older adapter.

The [transfer/accumulation report](../docs/experiments/results/curriculum-transfer-accumulation-20261121.md) records later-course ordering, replay/lineage checks and the explicit research-admission amendment. `tools/curriculum_transfer.py` retains the pinned bind-parent/group/exact-tensor checks. Its child config supplies a seal-bound parent manifest and exact child data-ready manifest; an unfinished/unqualified model cannot continue. Positive full parent loading and actual tensor warm start have now executed. `tools/curriculum_course_order.py` keeps complete original epochs with all declared strata and rejects tiny tails; it does not change the frozen stage1 recipes.

## Sealed calls-course continuation

The original strict gate remains the default. Our completed stable-v2 final uses the explicit research amendment bound before final games. Preserve the original pass/fail and all prerequisite successful exits. Run within scheduled GPU resources, using real local receipt paths:

```bash
python tools/curriculum_parent.py \
  --registered runs/curriculum-optimization-v2/fit-plan/registered.json \
  --evidence runs/curriculum-optimization-v2/final-evidence-private.json \
  --fit-exit "$FIT_EXIT_RECEIPT" --audit-exit "$AUDIT_EXIT_RECEIPT" \
  --child-data-ready runs/curriculum-optimization-v2/stage2-data/data-ready.json \
  --research-policy-binding runs/curriculum-optimization-v2/research-parent-policy-binding-private.json \
  --out "$PARENT_SEAL"
python tools/register_curriculum_course.py \
  --config configs/training/curriculum-stage2-calls-v1.json --parent-seal "$PARENT_SEAL"
python tools/train_curriculum_course.py \
  --registered runs/curriculum-stage2-calls-v1/fit-plan/registered.json
python tools/audit_curriculum_course.py \
  --registered runs/curriculum-stage2-calls-v1/fit-plan/registered.json \
  --out runs/curriculum-stage2-calls-v1/final-evidence-private.json
```

The research-policy binding must already have been created with the frozen policy and `tools/register_curriculum_research_parent.py` before either final584 game result, as documented in the amendment. Do not backdate it or use an intermediate checkpoint. Aggregate original controls are preserved in [evidence/curriculum-initial-final64.json](evidence/curriculum-initial-final64.json) and [evidence/practical-v8-riichi107.json](evidence/practical-v8-riichi107.json); they contain no raw games or per-case predictions. A new-host experiment with different bundle metadata must explicitly bind its own control identities/prospective policy and source before viewing final outcomes, rather than silently weakening this recorded experiment's hashes. Software checks, research admission and eventual independent strength confirmation remain distinct.

Course2 uses the same adapter/head, a new optimizer,20,237/4,721 expert/replay decisions, two epochs and324 updates.50 curriculum regression checks and the real preflight passed; actual initialization verifies all372 adapter/4 head tensors. The first16 native games after reload match the parent's same16 outcomes. Course2's loss mixture differs from stage1, so preserve separate curves. Stage3/4 and the unseen final comparison are not started by these commands.

## Recovered simulation provenance

The [stage3 source report](../docs/experiments/results/curriculum-stage3-provenance-20261121.md) supplements the public-data audit: all4,083 anonymized MC rows map uniquely to original complete inputs/targets, and all2,722 current informative TRAIN targets reproduce from the original paired-world artifacts. The code and three added checks are public. Complete per-world reconstruction needs the private original artifacts; fitting the unchanged public targets does not. The16/32-world conditional uncertainty, old fixed-opponent-hand sampler and hand-only horizon remain explicit limitations. No new raw hands, world logs, per-record source links, confidence details or weights are published. This audit changes no active fit and does not launch stage3/4.

The [attack-course preparation report](../docs/experiments/results/curriculum-attack-course-preparation-20261121.md) adds executable replay-preserving preparation and preflight commands. All22,959/5,905 informative rows keep their model inputs and labels. The75% expert replay component preserves the calls-course relative weights;25% MC is split equally across four phases. Actual API replay, ancestral group separation,160 loss/gradient comparisons and two complete sampler permutations passed. The new calls-final sealing bridge is separate from frozen earlier bridges; its real positive seal/load remains pending the actual stage2 final/audit. A prepared dataset and a sampler check are not stage3 training.
