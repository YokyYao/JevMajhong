# Terminal diagnosis: real decline has sampled top-two gain; small-win sources are sparse

2026-11-23, Asia/Dubai. Read-only GPU317477 completes0:0 in7s on ws-l1-002.
All nine saved plans/68 cached terminal-arm records reconstruct their actual
choices and top-two/rank/net utility components. The925-row canonical source
corpus is joined with all310 existing public win-score details. No model
inference, new game, training label, reward change or student update occurs.

## What drove the observed declines

| Saved plan cohort | Plans | Declines | Reconstructed reason |
|---|---:|---:|---|
| Initial four public contexts |4|0|All accept|
| Synthetic future fixture with real v8-backed planning |1|1|Sampled top-two and rank tie; net component alone favors decline|
| TRAIN-derived later opportunities |4|1|Decline has a positive sampled top-two difference|

For every arm, mean top2 +0.02*(4-rank)/3 +0.005*tanh(net/8000) reproduces the
saved utility within1e-12, as do paired component differences and selected keys.
Private per-plan component details stay private. These68 records were already
generated in the completed pilots; they are not68 new or independent games.

The synthetic example exposes a secondary-score preference when sampled
placement ties. It supports examining the tie behavior, not rewriting a reward
from a single fixture. The actual TRAIN-derived decline is not driven merely
by net points: it has a sampled top-two advantage. Nevertheless, its outer
continuation retained the same final own rank/net. Four sampled worlds do not
establish a stable advantage, and one realized continuation cannot validate or
refute an expected-value difference. More independent cases and sensitivity
evidence are needed before admitting labels. This candidate teacher has not
trained the student, so this diagnosis does not explain production-v8 or the
running calls student's behavior.

## Public small-win proxy coverage

The925 canonical records include310 win views. The existing narrow public
scorer supports295; the other15 are explicitly unsupported kan-hand special
cases, not assigned zero. These are cached no-ura, sole-winner immediate
transfers, not new scoring replays or guaranteed final placement.

Select a coverage proxy: current rank3/4, supported public solo income<=3000,
and immediate solo rank still>2. The actual coverage is:

| Public score context | TRAIN proxy views | Validation proxy views |
|---|---:|---:|
| Third, gap3000 |5|2|
| Fourth, gap5000 |7|4|
| Fourth, gap13000 |8|4|
| Total views across these contexts |20|10|
| Distinct underlying physical decisions |8|4|
| Distinct underlying source families |8|4|

The same eight/four physical decisions recur under different public scores;
twenty/ten views are not twenty/ten independent examples. Source-family
TRAIN/validation overlap is zero. In TRAIN, the proxy includes ordinary ron
and tsumo, mostly open hands and only zero/one-opponent-RI cases. For the
user's third-gap3000 concern there are only five TRAIN source decisions.

Leading rank1/2 small-win counterparts provide eight TRAIN/four validation
views in each of hold_first and hold_second, again from eight/four distinct
physical decisions/families across contexts. They are useful coverage controls,
not automatically labelled accept. A small win can still leave continuation,
and declining is not automatically correct in the trailing proxy population.

Unknown ura, simultaneous winners, payer, dealer repeats/extensions and future
play can change actual income and terminal rank. The proxy must not become a
rule that always rejects small wins. The earlier four-context pilot was selected
by public kind/context/ID, not this proxy, and cannot substitute for targeted
small-win coverage or reliable labels.

## Execution, publication and next action

Sourcefa7c9ff7a8ca37a64bff05c04b308032cbd01a1b has exact remote readback of
all three code/plan/entry files. The
[prospective audit plan](../plans/curriculum-terminal-objective-20261123.md)
fixes complete saved-plan/data populations before reconstruction. Actual prior
success receipts, input hashes, components and source splits are checked;
all224 active-fit dependencies remain unchanged. Source data, utility details
and eligible case IDs remain private. Only aggregate diagnosis is published.

The job uses one GPU/four companion CPUs/32GB with10-minute allocation,
8-minute foreground timeout, node/device check, logs and actual shell/scheduler
exit0 captured promptly. No login compute, CPU-only workaround, unmanaged
worker or background poller. Existing failed bound v1/v2 core tests remain
unchanged pending archival authorization; no overall-green CI claim.

The main calls fit316781 remains on its unchanged388-step recipe, with formal
validation/native evidence due at194/388 and independent audit316782. This
diagnosis does not change that training objective or select a midpoint.

Next examine reuse of the existing256 true-trajectory source matches to expand
terminal coverage beyond the current64-family cohort, preserving their actual
whole-family splits. The saved parent contains complete MJAI trajectories;
it does not record every unexecuted losing call intention. Any replay harvester
must reproduce native events/ledgers and must not turn inferred intentions
into expert labels. Retain ordinary progression/call as well as win windows
for the full placement course, rather than shrinking it to win/pass only.

Then freeze additional public-metadata-selected cases and more-world/seed/
continuation sensitivity before any target or teacher admission. Preserve all
outcomes, not only positive estimates. Actual risk/placement fits, real curves
and the unopened256-match-per-policy v8 comparison remain required. No new
weights/raw records, main merge, production change or Windows rebuild.
