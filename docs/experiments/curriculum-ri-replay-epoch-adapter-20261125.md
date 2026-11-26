# Weighted replay: epoch-to-original batch mapping

The candidate replay schedule now has an adapter for the pinned JevAny
training batch path. Each physical slot is resolved through its own epoch
plan before augmentation and encoding, and its original-row weight is attached
after encoding. This prevents the second epoch from accidentally reusing the
first epoch's row IDs, option-permutation seed or inverse-repeat weights.

| Completed integration check | Result |
|---|---:|
| Epochs checked |2|
| Physical batches through pinned `encode_batch` |180,456|
| Unique original TRAIN rows reached in each epoch |87,595 / 87,595|
| Source categories with exact planned counts |19 in each epoch|
| Wrong row/weight/observed-action mappings |0|
| Unregistered augmentation or skipped slots |0|
| Unit tests |4 passed|

The full pass uses the real frozen TRAIN rows and exposure plans, the pinned
trainer's actual augmentation, deterministic per-original/epoch RNG and
materializer, and the previously recorded real unique-row admission of all
87,595 rows. The model encoder is an explicit spy with stub IDs. No new full
token-admission pass, actual model forward/backward, optimizer update or game
is claimed. The separate loss preflight already checked real tokenizer
invariance for all 33 distinct source/weight combinations; this pass checks
complete dispatch and accounting.

The adapter expands only after unchanged unique-row admission succeeds. Its
virtual slots contain private metadata, and it resolves them to the actual
original request before the trainer derives augmentation RNG from epoch and
original ID. Option permutations therefore remain those of the original
decision. Both epochs retain the complete original corpus with their own
planned repetitions and weights.

It refuses missing or repeated positions, stale epochs, early epoch changes,
unregistered none/distractor augmentation, changed batch/accumulation settings,
incomplete unique-row admission and corrupt per-original total masses. It
records actual source counts, original-row coverage and mapping/weight hashes.
Only the global epoch-order RNG is intercepted; per-original augmentation RNGs
remain the pinned implementation. It attaches weights after `model.encode`,
so loss metadata is absent from that encoder invocation.

Four tests cover both epochs' independent mappings, original-data preservation,
admission-before-expansion, skipped/stale positions, unfinished epochs, changed
augmentation, RNG scope and invalid mass correction. The complete preflight
finished with exit0 inside the existing finite GPU allocation. It restores
its temporary hooks before exit. The running trainer is a separate process and
its source, configuration, optimizer and weights remain unchanged.

The candidate still needs a distinct fit registration, explicit continuation
checkpoint and optimizer schedule, real-model warm-start/integration evidence,
native evaluation and independent audit. No candidate parent was selected and
no fit started. The current retained-parent trial continues its original
2,400-update budget; production v8 and the reserved final grid remain unchanged.

Inside a finite approved GPU allocation, reproduce this integration check after
the earlier replay/loss preparations:

```bash
python -m unittest tests.core.test_curriculum_ri_replay_epoch_adapter_v1
python tools/curriculum_ri_replay_epoch_adapter_v1.py
```

Only source, tests and this aggregate report are published. Raw requests,
virtual indices, per-original weights and detailed mapping hashes stay private.
