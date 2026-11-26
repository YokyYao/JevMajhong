# Calls194 RI diagnosis: lower total loss does not preserve RI discrimination

2026-11-23, Asia/Dubai. GPU317827 on ws-l1-002 completes0:0 in7s. Two
independent hand-calculated numerical checks pass. The audit uses only saved
calls0/194 probabilities on the unchanged632 basic RI validation windows,
verifies all fixed0.5 source confusions against the actual validation journal,
and changes no threshold, weight, game or student update.

| Same cached RI windows | Step0 | Step194 |
|---|---:|---:|
| Pooled ROC-AUC |0.7327|0.7217|
| Pooled Brier diagnostic |0.2097|0.2143|
| Fixed0.5 balanced accuracy |69.27%|67.34%|
| Positive targets selected |208/324|226/324|
| Negative targets correctly declined |229/308|200/308|
| Mean predicted RI mass |47.80%|50.08%|
| Ten-bin descriptive ECE |0.0743|0.0601|

Observed accept fraction is unchanged at324/632=51.27%. The lower ECE alongside
worse Brier/AUC/decline recall shows why one pooled metric cannot establish a
solution. These outputs are action preferences, not calibrated acceptance or
win probabilities. Source/class weighting intentionally changes effective
priors, so calibration-to-observed-label diagnostics are not automatically the
training objective or a reason to fit a new threshold.

| Source | Windows / source groups | AUC0→194 | Fixed0.5 balanced accuracy0→194 |
|---|---:|---:|---:|
| Legacy |107 /74|0.7432→0.7266|70.14%→69.19%|
| New teacher |525 /51|0.7345→0.7227|69.52%→67.51%|
| Legacy, no opponent RI |84 /66|0.7296→0.6946|68.27%→64.44%|
| Legacy, opponent RI |23 /18|0.5132→0.5789|59.87%→72.37%|
| Teacher, no opponent RI |341 /51|0.7295→0.7063|67.03%→63.95%|
| Teacher, opponent RI |184 /44|0.6095→0.5932|61.58%→59.37%|

Groups overlap between threat/no-threat strata; these rows are not independent
populations. Legacy threat windows contain only four positive targets; their
improved point estimates do not establish a general threat-handling gain.
Teacher threat windows contain66 actual RI choices and118 declines;
opponent RI is not an automatic expert instruction to decline. At194 the
model selects23/66 positives but also19/118 negatives. In teacher no-threat
windows it selects174/221 positives and61/120 negatives. Legacy no-threat
false positives also rise20→27, so deterioration is not confined to defense
against an existing opponent RI.

Point-estimate AUC falls in both sources. A monotone recalibration preserves
ranking and cannot by itself restore a lost ordering; this does not prove a
statistically significant ranking deterioration or rule out useful future
training. No source-cluster confidence interval is estimated here, and a
different operating point might change precision/recall tradeoffs. Do not
present this consumed-validation diagnostic as policy strength or tune a
threshold/checkpoint to it.

The findings support checking shared-task interference, class-versus-pointer
loss alignment and effective update/data budgets at the completed388 result,
instead of assuming that lower overall CE or more repetitions of the same mix
has solved RI decisions. The current max-discard class loss and inference
full-class0.5 boundary are distinct by design. Simply replacing max class CE
with group CE plus the same conditional-discard CE reduces algebraically to
ordinary full-action CE for these one-reach menus; that is not automatically a
new repair, and prior hard-CE controls must not be forgotten. A subsequent
change needs its own frozen comparison and native games.

[The prospective source](../plans/curriculum-calls-riichi-calibration-20261123.md)
is460389faf5a01b0eddafb2f3f40debac0339048b with four exact GitHub readbacks.
The actual0/194 validation confusions match, and all224 active-fit dependencies
remain unchanged. Saved predictions, individual labels and reliability details
remain private; only aggregate diagnostics and code/plan are public. No new
inference, games, calibration fitting or optimizer update is run. Actual
scheduler/shell exits are saved. All processing/hashing/tests use finite GPU
allocations with companion CPUs, resource checks, logs and timeouts; no login
compute is used. The frozen388 fit and final audit continue, and the unopened
final v8 comparison remains required.
