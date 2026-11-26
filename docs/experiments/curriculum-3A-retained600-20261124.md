# Retained-replay3A: first600-update development milestone

The retained-parent trial has completed600 optimizer updates, all23688
validation menus and its scheduled16 complete East development matches.
The parent is the qualified calls388 model under complete-legal execution;
this trial continues its weights with a new optimizer. Its frozen87595-row
TRAIN corpus, paired order, source masses and full2400-update budget are
unchanged. No intermediate checkpoint is promoted or admitted to3B.

The replay-loss change reduces the RI drift seen in the paired Class2 control,
but does not eliminate it relative to the parent. Correct decline means
agreement with held-out labels, not proof that every label is optimal.

| Full validation diagnostic | Parent0 | Paired Class2 control600 | Retained600 |
|---|---:|---:|---:|
| Legacy RI correct decline |50/70 ·71.43%|30/70 ·42.86%|43/70 ·61.43%|
| Teacher RI correct decline |176/238 ·73.95%|104/238 ·43.70%|158/238 ·66.39%|
| Legacy RI balanced accuracy |70.85%|64.67%|67.20%|
| Teacher RI balanced accuracy |70.42%|62.79%|68.04%|
| Canonical weighted target CE |1.136669|1.116347|1.110398|

Retained600 RI acceptance is27/37 legacy and200/287 teacher. Both accept/
decline source floors remain above60%, but legacy balanced accuracy drops
3.65 percentage points from its parent, exceeding the original3-point
retention allowance. The basic-retention diagnostic therefore fails at this
milestone. Plain-label agreement gains0.05 points legacy/1.17 points teacher;
mean five-call balanced accuracy gains0.60 points and equal-phase masked-MC
CE improves0.10715. These label diagnostics are separate from native strength.

| Exact same16 development cells | First | Top two | Net points/match | Hand-win rate | Deal-in rate |
|---|---:|---:|---:|---:|---:|
| Parent0 complete-legal policy |31.25%|56.25%|+387.50|17.75%|7.69%|
| Paired Class2 control600 |25.00%|56.25%|−62.50|16.57%|6.36%|
| Retained600 |18.75%|62.50%|−875.00|16.72%|4.65%|
| Frozen practical v8 |62.50%|62.50%|+3612.50|22.37%|10.34%|

The new policy gains one top-two result versus its parent, while net points
fall1262.50/match and first-place results decline. Lower deal-in alone is not
an overall improvement. These are only two consumed logical seeds crossed
with four seats/two profiles; they do not establish stable gains or v8 parity.
Hand-win/deal-in rates are per-match hand ratios averaged across matches.
All64 compared native match ledgers/counters were independently reconstructed
without loading a model or generating another match. Validation prefixes,
model identities and actual source files are preserved privately. Retained600
checkpoint bundle: `7c56b1622f9cbfdaaf63e7541d6cb8e649df0c781be78a9b11ff32ae6d7370cd`.

Continue the registered full trial and inspect1200/1800/final2400. The current
results support preserving the inherited formula to reduce supervised RI
collapse; they do not demonstrate a native net-point improvement. Final
original qualification and the independent audit remain mandatory. The
unopened final comparison, main and v8/Windows remain unchanged. Raw records,
weights and simulation traces are not published.
