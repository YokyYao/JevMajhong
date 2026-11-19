# Fourth-place pursuit: balanced continuation of the existing LoRA

The user authorized analysis of the two-update stop and revised continued
training. The intervening full-source AdamW head experiment completed128
updates, not an early stop. Training strict fourth targets improved14→30/48,
and validation soft top2 .1829968→.1917080, while greedy finite-label top2
.2230392→.2205882 did not improve. Old replay CE and some reference KLs also
regressed. This does not support automatic extension of the same head fit.
The ordinary-fourth strict gradient has cosine−.776 with ordinary replay and
−.826 with leading replay at initialization. This is measured evidence of
source conflict; it is not proof that the encoder is the only bottleneck.

This next experiment changes representation learning and exposure together.
It therefore cannot attribute gains to either change alone. It retains all
round16 data/targets, final-top2 fourth reward, source masses and selection
guards. Fourth escape and native-fourth KL remain reporting-only.

- Warm-start original released Play v2/v8 step2088 LoRA AND head, not the
  rejected128-step head. Train its existing rank16 LoRA and524,800-parameter
  head; base weights stay frozen. No extra residual network or inference pass.
- Reuse the same905 train/186 validation membership,826 positive rows and149
  fourth training targets. No new labels/simulations or train/validation mix.
  Reuse tokens only; compute fresh encoder activations and gradients.
- Each32-record gradient accumulation has16 ordinary replay,4 leading replay,
  4 fourth win/continue,2 third win/continue,2 third ordinary,2 holding,
  1 strict ordinary fourth and1 other ordinary fourth. Each source cycles
  without replacement until exhausted. All826 unique positive rows appear
  within39 updates. The fixed64-update observation window covers all data.
- Each row's loss coefficient is its fixed source mass divided by that
  source's batch count. Thus stratified sampling does not increase fourth's
  .35 total objective mass or erase the .50 replay retention objective.
- AdamW LoRA LR2e-6, head LR1e-5 (five times lower than the longer-head trial),
  warmup8 then constant, decay.001, clip1. BF16 backbone/forward with FP32
  adapters/head. Gradient checkpointing. No per-source monotonic CE veto.
- At most128 updates. Full186 validation plus149 fourth training diagnostics
  every16, including zero. First64 observe; afterward three stale validation
  checks stop (earliest112). Soft top2 gain>1e-5 or hard best resets patience,
  but selection still requires actual hard top2 gain and original third,
  leading and replay protections AFTER all826 examples have been exposed.
  Nonfinite loss/gradients abort; replay CE>formal anchor+.50 stops separately.
- Keep formal-v8 validation anchors. If live unmerged numerics give a better
  zero hard top2, use that higher floor too; never lower the formal anchor.
  Actual saved-and-loaded inference must independently reproduce qualification
  before consumed complete continuations. No new independent/full-start or
  release until development qualifies. A soft metric is not actual win rate.

Require the completed128-step head saved-weight/service audit before this
fit starts. Initialization verifies exact formal-v8 tensors, every validation
saves adapter/head/optimizer/RNG/exposure, and checks prove both original
LoRA and head actually change. Source/data fingerprints remain frozen.

Run only inside bounded Slurm GPU allocation with companion CPUs, logs and
exit receipt, foreground child handling and scheduler time limits. Public
content is code, plans and aggregate summaries. Raw traces, caches and
research weights remain private. Main and Windows0.4.2 are unchanged.
