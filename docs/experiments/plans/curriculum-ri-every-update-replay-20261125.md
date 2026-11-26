# Prospective replay arrangement for steadier RI updates

The current retained-parent trial has balanced accept/decline source masses,
but its rare legacy RI sources occur in only192/260 of the first1200 updates.
The paired-order control already pairs samples within511-record mixing blocks;
that does not ensure each73-record optimizer update sees both target classes.
Test a sampling-variance hypothesis before increasing the class masses again.

Keep every original TRAIN decision, target and19-source loss mass. Preserve the
baseline non-RI order. Every candidate73-record update contains one example from
each of the four inherited RI strata and69 non-RI examples. Cycle through each
RI source in deterministic shuffled passes. Insert only the minimal non-RI
repeats needed for full updates, distributed throughout the original order.

For original decision i in source s with original count N_s and total repetition
c_i, assign each appearance weight E * mass_s / (N_s * c_i), where E is the
candidate epoch's physical exposure count. Thus the total normalized weight
of every original decision remains exactly mass_s / N_s. This preserves the
original weighted empirical objective at fixed parameters, including its full
gradient, while changing optimizer update order and variance. It does not prove
equivalent optimization trajectories or a stronger policy.

Prepare both epochs from the actually registered baseline permutations. Check
complete unique-row coverage, absence of validation-family overlap, positive
weights within16, per-epoch repeat cap8 and exact per-original total mass.
Compare the actual per-update effective RI mass distribution against the old
schedule. Inverse-repeat weights can vary between updates: do not claim exact
per-update class-mass equality merely because every class is present.

This is data-arrangement preparation only. The live2400-update trial, current
optimizer and final-only advancement remain unchanged. The candidate needs a
distinct weighted-loss/journal integration, frozen initialization and numerical
budget before a future fit. Select the continuation checkpoint from development
evidence after the current trial and TRAIN-only gradient diagnosis; no new base
restart or automatic parent admission is implied. Retain the existing baseline
and require actual RI retention and native-score evidence from the next trial.
The independent final comparison remains unopened.

Run preparation and tests only inside finite scheduled GPU resources. Publish
source, plan and aggregate exposure/variance results; keep original decisions,
indices, per-record weights and all raw records private.
