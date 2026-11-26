# TRAIN-only RI task-gradient diagnosis

The retained-parent Stage 3A1200 model raises reach preference on both RI target
classes despite balanced inherited accept/decline loss masses. A full corpus
scan rules out omitted decline exposures and reach-positive hard labels in the
two new hard-label teacher strata. It does not establish which parameter updates
cause the drift. Measure actual local gradients before choosing the next recipe.

Freeze at most eight deterministic distinct TRAIN families per each of the19
registered sources, independent of losses, validation predictions or game
outcomes. Use the same samples on the saved complete-legal parent0 and actual
retained1200 model. Read validation metadata only to exclude cross-split families;
never compute validation gradients or insert held-out examples into TRAIN.

For every source, average the actual current retained-parent objective gradients
over all372 LoRA/four pointer-head tensors. Apply the original source loss masses
when combining them. Report decline-versus-new, decline-versus-accept and
decline-versus-total dot products/cosines for all parameters, head and LoRA
separately. Two fixed alternating family blocks reveal whether the sampled sign
is consistent. The sample is small and is not an estimate of the entire corpus.

Run in eval mode with dropout disabled and without an optimizer. Negative dot
products predict first-order loss conflict under simple negative-gradient
steps only; they do not reproduce Adam moments, establish causal training
trajectories or prove stronger play. No model is selected or updated. Check
every trainable parameter in memory and the saved checkpoint identity before
and after the probe. Preserve complete logs, finite timeout and actual exit.

Prepare the frozen TRAIN selection inside the existing GPU allocation. Queue
the separate one-hour GPU probe after the current fit/audit terminal-capture
job, at low scheduling priority; do not compete with the live training process
for GPU memory. No polling loop or automatic next training is introduced. A
failed or interrupted attempt keeps its private output and requires a new
version for another attempt.

Use the resulting evidence to decide a prospective change to replay retention,
objective or mixture. Do not change the current2400-update trial mid-run, alter
its advancement rules, tune a serving threshold against validation, or open
the reserved final comparison. Publish source, this plan and aggregate findings
only; selections, per-family details and gradient records remain private.
