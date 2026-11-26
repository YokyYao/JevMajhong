# Stage3B: real first-hand continuations add3,969 teacher decisions

All512 previously constructed source cases now reach their first actual hand
settlement under four stateful Mortal players. The collector adds3,232 TRAIN
and737 validation observer0 decisions, excluding the already recorded initial
windows and rule-forced win choices. Initial plus continuation gives3,652 TRAIN
and828 validation raw teacher decisions before normalization/deduplication.
These are still the same246 inherited source families, not additional
independent source games or student training results.

| Later observer decision | TRAIN | Validation |
|---|---:|---:|
| Closed discard windows |2,095|430|
| Open discard windows |536|151|
| Chi/pon windows |538|128|
| Kan windows |63|28|
| With one opponent riichi |589|161|
| With two opponent riichi |85|14|

The threat rows overlap the decision-phase rows; do not sum them as additional
examples. Closed ankan remains distinct from open melds. Decisions range from
early play to the observer's19th discard-number window, including response and
post-call contexts. Eight TRAIN rows follow the observer's own riichi.

## Actual source wins include rare hands and alternative routes

These are supported public no-ura yaku counts within each intended source
cohort. They describe teacher/source outcomes on deliberately biased deals,
not a student ability, natural-game rate or certified optimal action label.

| Intended source cohort and supported target yaku | TRAIN | Validation |
|---|---:|---:|
| Kokushi, including thirteen-sided variant |28|4|
| Chuuren, including pure/double variant |11|1|
| Suuankou |8|5|
| Daisangen |18|5|
| Chiitoitsu comparison cohort |21|2|

Other source wins turn into ordinary yaku such as chinitsu, honitsu, yakuhai,
sanankou and toitoi. All failed pursuits, aborts, other-player wins, observer
deal-ins and draws remain in the cohort. An intended route tag does not assert
that the player actually pursued, completed or should have pursued that route.
The supported yaku screen omits explicit kan/first-turn exceptions; they are
reported as unsupported, never assigned an invented yaku. Native event ledgers
conserve points/sticks but are not a second full han/fu scoring engine.

| Selected later call | TRAIN | Validation |
|---|---:|---:|
| Chi |54|10|
| Pon |83|22|
| Ankan |13|1|
| Kakan |4|1|
| Daiminkan |0|0|

Daiminkan is offered17/6 times but never chosen in this cohort. This does not
prove all offers were bad or repair the existing daiminkan training gap. Retain
the old calls replay and consider separate action-value sources where needed.

## Evidence and reuse

GPU318683 on ws-l1-002 completes0:0 in3m31s, with22,952 actual native action
batches. Every source reuses its actual full wall, initial public request and
family split. All511 original network first choices reproduce exactly; the
single initial forced win stays a rule outcome. Fresh Bots are initialized per
actual environment and follow its genuine public history. No actor is reused
after hidden-hand resampling. Simultaneous teacher intentions are recorded
before the engine resolves priorities; a missing executed call is not a fake
PASS label.

The complete first-hand traces/logs and financial ledgers are stored privately
once per source for later reconstruction. Model-visible inputs contain no
target route, hidden hand, future event or completed-yaku annotation. All
partitions remain inherited; no source is moved to create a desired result.
The currently running3A data/weights are untouched.

The reusable normalized corpus is now91,247 TRAIN/3,161 families and24,516
validation/975 families. It retains all87,595/23,688 original replay objects
and adds all3,652/828 new teacher decisions under six private route-source
tags. All4,480 new cases encode without truncation or label/menu changes;
maximum new packed lengths are1,319 TRAIN and1,125 validation. No duplicates,
conflicting labels or cross-partition public inputs require quarantine.
Existing replay audits are reused, not repeated, and no backbone is loaded.

Preparation318689 fails on a wrong packed encoder field before admitting any
data; its source and failure are retained. Corrected318695 succeeds with shell
exit0, using question keys from the materialized record and separately checking
actual packed labels/options. It reuses the collected sources, not another
teacher run. [Aggregate normalized coverage](curriculum-yaku-data-20261123.json)
contains source/route/phase support and actual token maxima. A3B loss mix,
parent and student fit still require separate registration after actual3A results.

[Collection plan](../plans/curriculum-yaku-hands-20261123.md) ·
[Data preparation and narrow v2 fix](../plans/curriculum-yaku-data-v2-20261123.md) ·
[Initial source cohort](curriculum-yaku-source-pilot-20261123.md)
