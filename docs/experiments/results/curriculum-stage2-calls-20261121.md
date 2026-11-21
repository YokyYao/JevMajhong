# Calls course: interim162 result

2026-11-21, Asia/Dubai. The registered calls course is still running as GPU312081, with final audit312082 after successful completion. The sole candidate remains final324. This report contains the completed162-point observation, not final promotion or an independent v8-strength test. Configuration, data and loss masses remain unchanged.

## Same16 complete East matches

The grid is two consumed development seeds ×four own seats ×two opponent profiles. The initial student's parameters exactly equal the completed fresh-base stable-v2 parent; its16 outcomes reproduce the parent's same subset. Full64 results are not mixed with these16-game points.

| Metric | Parent / step0 | Calls step162 | Frozen v8 |
|---|---:|---:|---:|
| First |25.00%|56.25%|62.50%|
| Top-two |62.50%|62.50%|62.50%|
| Fourth |18.75%|18.75%|31.25%|
| Net points/match |−1,193.75|+2,950.00|+3,612.50|
| Hand-win |20.07%|21.69%|22.37%|
| Dealin |12.86%|15.51%|10.34%|
| Observed net/8000 reward |−0.1492|+0.3688|+0.4516|

Net increases4,143.75 points/match over the parent, while top-two is unchanged and dealin increases2.65 percentage points. Against v8 on this same subset, net remains662.50 lower and dealin is about5.17 points higher. Two whole-seed groups are too few for a stable strength conclusion; these means neither establish equivalence nor select this checkpoint. Continue the fixed budget and evaluate final324 on all64 registered games.

Actual own-operation totals across the corresponding16 games are below. They are **counts, not opportunity-conditioned selection rates**; policies change subsequent hands and available decisions.

| Operation | Parent / step0 | Step162 | v8 |
|---|---:|---:|---:|
| Chi |53|41|18|
| Pon |50|50|37|
| Ankan |3|5|4|
| Daiminkan |0|6|2|
| Kakan |0|0|3|
| Riichi |31|14|20|

The student takes daiminkan in real simulator games, and net improves without increasing the chi count. Neither fact proves the chosen calls are optimal or isolates the cause of the score change. Zero kakan in16 games does not establish inability without opportunity counts; held-out menus below show actual kakan selections.

## Fixed4,721-menu validation

Weighted CE changes1.116720 at0 →1.023091 at81 →1.012235 at162. Ordinary exact agreement is50.72% →50.35% →50.56%. Riichi swings between acceptance and decline: TP/FN/TN/FP is26/11/30/40 initially,35/2/10/60 at81, and17/20/48/22 at162. Balanced type agreement at162 is about57.26%, but neither acceptance nor decline is reliably solved by these changing endpoints.

At162, operation-specific counts include overlapping menus. A nonselected operation can lose to another operation, not only to pass.

| Operation | TP | FN | TN | FP |
|---|---:|---:|---:|---:|
| Chi |141|70|320|74|
| Pon |165|50|285|86|
| Ankan |95|41|64|69|
| Daiminkan |88|0|3|105|
| Kakan |34|23|22|24|

**Daiminkan declining is a clear unresolved diagnostic:** all88 observed positives are selected, but only3/108 observed negatives avoid it. The aggregate five-call balanced score can hide that behavior. These labels are observed expert decisions, not optimality certificates; still, the result warrants checking type-specific counterexamples and the frozen v8 diagnostic before claiming calls are solved. Preserve the current fit. Use its final result and data/rollout coverage to decide a separately registered correction rather than silently changing active weights or thresholds.

Next checkpoints are validation243, then final324 validation/64 games/full audit. The [attack recipe](../plans/curriculum-stage3-attack-20261121.md) is implemented and prechecked but not registered or trained. It still needs an actual qualified final calls parent. Research weights, raw game logs, per-menu predictions and diagnostics remain private; only these aggregate observations are published.
