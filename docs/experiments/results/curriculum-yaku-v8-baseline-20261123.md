# Frozen v8 already completes rare yaku in the new conditional source cases

The fixed92-case validation benchmark is complete against three stateful
Mortal opponents per hand. Frozen v8 keeps its original PolicyAgent route and
exact parameter identity5ebce6af3777854cbec55ba29db2dc90be0f959193114bb3154179c5552370fa.
It reaches actual kokushi, chuuren, suuankou and daisangen wins after making
earlier decisions. Therefore subsequent3B work must assess retention and
improvement against existing ability, not describe every rare hand as a new
capability introduced from zero.

These deliberately biased source deals are single-hand conditional episodes,
not complete East matches or natural-game Mahjong Soul win rates. In particular,
their net transfers cannot be compared with v8's+4,290.63 points/complete East
match on the separate established64-match development grid, which was not rerun.

| Intended source cohort | Cases / families | Observer wins | Deal-ins | First-hand mean net | Supported actual target yaku wins |
|---|---:|---:|---:|---:|---:|
| Kokushi |14 /6|5|1|+12,992.86|4|
| Chuuren |16 /8|6|1|+8,125.00|1|
| Suuankou |22 /11|11|2|+11,009.09|3|
| Daisangen |16 /8|8|4|+21,081.25|3|
| Chiitoitsu comparison |20 /10|6|3|+2,430.00|3|
| Random-initial-hand control |4 /2|1|1|−350.00|not applicable|

The daisangen cohort includes one initial legal-win shortcut, identical by
construction across policies; it is reported separately and is not evidence
of learned hand construction. Supported target-yaku counts exclude explicit
kan/first-turn scorer limitations. Other wins may have different yaku, and
unscored wins are not assumed to lack the target yaku. The policy's rule guard
still executes legal wins: the evidence here concerns earlier route decisions,
not whether the network itself recognizes or scores a completed hand.

V8 also actually selects daiminkan three times, in the daisangen and chiitoitsu
source cohorts. This is evidence that the deployed policy can select that
operation in some legal contexts, not proof those choices improve expected
value or that every important kan context is solved. Frozen source counts,
all actual action traces and point/stick ledgers are retained privately.

## Cached comparator and next training

All92 original validation cases/45 source families are retained with their
exact full walls and public initial histories. No source is selected by its
winning outcome. Other players get fresh stateful teacher actors following
the new actual public history; there is no hidden-hand resampling with stale
actors. Target-route annotations, concealed opponents and future events never
enter the policy request. Existing all-Mortal source outcomes are descriptive,
not a formally matched comparator with implicitly identical numerical settings.

Use this cached frozen-v8 result for future matched candidate comparisons
under the same code/walls/opponent setup. The source-case cohort is development
data, not independent final confirmation or a substitute for natural complete
matches. Small per-route family counts limit conclusions; preserve the unopened
final East-match grid.

The [candidate3B recipe](../plans/curriculum-yaku-course-20261123.md) and
[actual TRAIN-only preparation metadata](curriculum-yaku-recipe-20261123.json)
reuse the old loss, propose80% replay/20% equal new-route loss mass, and prepare
two complete epochs/2,500 total updates/182,494 exposures without replacement. This is
not a registered parent or active3B fit. Current3A training continues unchanged;
actual3A results precede the3B parent/fit decision.

GPU318747 on ws-l1-002 completes0:0 in1m20s. The benchmark checks all actual
first-hand ledgers and verifies that the reference parameter bundle and bound
inputs remain unchanged. Raw case traces stay private. Main, Windows, the
production model and current3A training are unchanged.

[Aggregate92-case results](curriculum-yaku-v8-baseline-20261123.json) ·
[Benchmark protocol](../plans/curriculum-yaku-validation-cases-20261123.md) ·
[New teacher source data](curriculum-yaku-hands-20261123.md)
