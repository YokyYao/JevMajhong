# Curriculum stage1 reproduction

This is the newly registered basic-discard/riichi course, not a replay of every historical v8 experiment. The student initializes a fresh rank16 LoRA and256-dimensional action head on pinned Qwen3.5-0.8B. No v7/v8/Mortal weights initialize it. The frozen v8 export is used only by a separate evaluation process.

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
python tools/plot_curriculum.py \
  --metrics-dir runs/curriculum-reproduction-v1/stage1-fit/metrics
```

The initial model is a random Mahjong head/adapter on a pretrained backbone, not a random-action agent; the legal-action/efficiency guard and forced legal wins still apply. Do not call its initial game results a learned Mahjong ability. Stage1 has no win/pass or yaku-scoring supervision. Stage2/3 data are prepared with the same holdout exclusions; their fits and stage4's public data generation are separate registered work and are not automatically launched by this command.

Raw match events and new weights stay local. Only aggregate results/curves are intended for publication. Existing v8 parameters and the Windows tool are unchanged.
