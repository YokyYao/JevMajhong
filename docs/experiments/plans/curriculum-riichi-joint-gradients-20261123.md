# Frozen-checkpoint ordinary/RI gradient geometry, without training updates

The current Qwen152 policy has poor matched16 games and weak RI refusal while
an offline public-context probe finds accessible binary-label signal. Joint
ordinary/RI gradient interaction remains unmeasured: earlier loss shares and
logit-gradient prototypes are not actual LoRA/head gradients. Do not change or
stop the running608 control based on this diagnostic.

Before calculation, freeze the current class-control checkpoints0 and152, its
unchanged TRAIN data and source masses. Hash-rank TRAIN IDs with seed2026112306
within each of six source/classes. Take the first eight distinct groups/source
and partition them into two panels of four/source (48 TRAIN cases/checkpoint,
96 example-checkpoint pairs). Check source-held-out exclusions. Use no validation
or final-grid cases to select panels or determine geometry.

Load each saved model in a bounded GPU job. Keep base parameters frozen;
enable gradients only for the372 existing LoRA and four pointer tensors. Use
dropout-disabled eval mode with bf16 forward/fp32 trainable tensors. Compute
ordinary CE, RI max-class CE and RI-decline conditional-discard CE parameter
gradients separately. Average within source/panel and combine with registered
loss masses. Report task norms/cosines for all trainable parameters, LoRA only
and pointer only. Distinguish zero-norm/undefined cosines from orthogonality.
Save per-case details privately; publish aggregate geometry only if useful.

Construct no optimizer and perform no parameter update or model save. Verify
trainable tensors unchanged, base parameters remain frozen, grad buffers remain
empty, actual input admission, gradient finiteness and gradient-mode/no-grad
forward canaries within5e-5. Preserve all source/data/checkpoint file hashes.

These are local Euclidean gradients on two small source-conditional TRAIN
panels, not actual stochastic94-record AdamW updates. Dropout, clipping,
optimizer moments/preconditioning and sampling differ. Negative cosine is
opposition for this local measurement, not proof of harmful optimization,
causality, persistent drift or a reason to project gradients/change source
weights. Report panel disagreement and parameter-block differences. Native
policy strength still needs actual games and complete final evidence; no
parent qualification, later course or production change follows from geometry.
