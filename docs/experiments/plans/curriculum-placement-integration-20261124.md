# Placement course: real-model integration before the full fit

Run `cluster/curriculum-placement-preflight-v1.slurm` only after the actual
qualified parent and budget registration succeed. This is a finite GPU job
with40-minute wall limit, scheduler log, exit receipt and timeout cleanup.
No integration computation runs on the login node.

The control loads the same pinned Qwen base and exactly the qualified final3B
LoRA/head tensors. It invokes the actual pinned JevAny trainer, not a mocked
optimizer or cached encoder. All57 sources contribute one pristine current
TRAIN-file record to the fitting control and a different semantic input to
the validation control. Neither control opens the real VAL file. These control
splits are not a strength test and are not added to student curves.

The actual production epoch adapter executes and verifies both complete
prepared epoch receipts, including all per-update source counts. The control
then restricts each full permutation to its57 selected TRAIN indices. It runs
two epochs, one accumulation group per epoch: two optimizer steps and114 raw
record exposures. This deliberately differs from the full course's exposure
distribution and accumulation; it does not test convergence at the full budget.

Checks cover exact inherited adapter/head tensors before updating, actual
weighted component losses versus upstream optimizer logs, LoRA and head
parameter changes, complete376-state new optimizer saves at steps1/2,
validation-before-save/deferred-callback ordering and the final callback without
a subsequent batch. The sealed parent bundle and every registered input must
remain unchanged. Control weights stay in a separate private directory and
cannot become the student candidate or enter its parameter ancestry.

The result is an integration proof, not permission to fit or release. The
complete launch binding and frozen development acceptance still need to bind
this proof together with the trainer, native evaluator and independent course
auditor. Full-course initialization must separately reproduce early input
probabilities, then run the changed child policy's fresh64-game zero baseline
before its first update. The final independent v8 comparison remains unopened.

Until this GPU control actually runs successfully, unit-test or CI success
must not be described as successful real-model integration.
