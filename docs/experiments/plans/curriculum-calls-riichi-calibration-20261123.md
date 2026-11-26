# Cached-only RI ranking/calibration diagnosis, calls0/194

The calls194 weighted validation loss improves while selected basic RI decline
recall falls74.35%→64.94%. Distinguish poorer score ranking from probability
calibration/route effects before proposing a subsequent fit. Keep the current
388 recipe, fixed0.5 selector and sole final candidate unchanged.

Use only the immutable saved predictions for the original632 basic RI windows,
with legacy107 and teacher525 kept separate. Sum full legal reach probability
mass; verify the fixed0.5 source confusion counts equal actual cached0/194
validation. Report pairwise ROC-AUC with exact ties, Brier loss, observed versus
mean predicted acceptance, fixed-threshold counts and ten equal-width reliability
bins. Also split each source by zero/nonzero opponent RI, retaining exact
denominators and source-group counts. Sparse/one-class strata retain null AUC
or balanced accuracy rather than inventing a metric.

Two explicit numerical checks use a hand-calculated mixed/tied example and
a one-class example, verifying AUC, Brier, fixed counts and reliability bins.

Do not scan thresholds, fit calibration parameters, choose a checkpoint, rerun
inference/games or update student weights. AUC/bins are descriptive on consumed
validation, not population confidence or policy-strength evidence. Cached RI
selections are not a replay of every native guarded discard. Bind source/data/
predictions and verify actual fit dependencies stay unchanged. Run inside a
finite10-minute GPU allocation with companion CPUs, node/device checks, timeout,
logs and exits. Raw per-record probabilities/labels stay private; publish only
code/plan and aggregate results after actual completion.
