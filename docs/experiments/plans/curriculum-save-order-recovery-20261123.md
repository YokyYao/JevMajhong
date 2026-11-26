# Save-order recovery and stage3 continuation

## Observed failure and preserved evidence

The first stage3A attempt (job318606) completed600 optimizer updates and its full23688-row validation, then failed because the validation tracker requested native evaluation before upstream saved that checkpoint. Only step0 weights exist; there is no saved600 model or optimizer. All failed artifacts remain private and unchanged. The600 discarded updates are excluded from the effective candidate ancestry.

Two isolated recovery preparations also failed before candidate updates: job318957 correctly rejected overlapping control train/evaluation samples; job318969 exposed the control tracker's missing offline config attribute. These failures and their sources remain unchanged. Neither updated a candidate or parent weight.

## Actual recovery route

GPU job318979 uses `cluster/curriculum-progression-v4.slurm`, an isolated v4 registration/output root, `tools/prepare_curriculum_progression_replay_v4.py` and the immutable `tools/train_curriculum_progression_v3.py`. The prepare function uses four semantically distinct TRAIN-only fixtures split2/2, a complete offline tracker interface and two real optimizer updates to exercise upstream validation→save→deferred native callback. This control fit is separate from the candidate process and never contributes to validation or reward curves. Existing step0 measurements may only be reused after exact loaded tensor identity and unchanged data/runtime checks; each post-update validation/game point is newly measured.

Stage3A retains the original sealed stage2 step388 parent, dataset membership, without-replacement epoch orders, seed2026112304, rank16, dropout0.05, learning rates and175190 exposures/2400 updates. Only checkpoint cadence changes to100 and the native callback waits until the complete model AND optimizer checkpoint is durable. A dedicated GPU final audit job318980 depends on actual success and retains all original quality criteria. The unused audit branch in the preserved v3 launcher is not a recovery entry point.

## Stage3B continuation

The new v3 3B launcher uses `configs/training/curriculum-yaku-fit-v2.json` and can only seal the actual audited final2400 model from job318979/318980 after the original3A advancement criteria pass. It retains the existing91247 TRAIN/24516 VAL corpus, six targeted yaku sources, exact replay, prepared source weights/orders, seed2026112305,182494 exposures and2500 updates. No peak checkpoint selection or gate waiver is introduced. No duplicate network is added.

The same checkpoint fence defers post-update native evaluation. Checkpoints are saved every125 updates, which divides every625-step validation point; saving every100 here would leave625 unsaved and recreate a failure. Step0 warm-start identity/prediction/game checks remain before the first candidate update; final2500 callback is flushed even without another batch. Subsequent dependent work is canceled on failed prerequisites rather than silently training from an unqualified model.

Only source, plans and aggregate status may be published. Raw rows, predictions, walls, event traces and research weights remain private. No main, Windows or production replacement is authorized by this recovery.

