# Stage3B source pilot:512 native decisions and511 actual teacher labels

During the independent running stage3A fit, the first targeted yaku source
pilot produces512 real native initial decision windows. They have420 TRAIN
cases/201 source families and92 validation cases/45 families. Replaying the
actual public requests/prefixes and querying the pinned Mortal-298k gives420
TRAIN and91 validation network-action labels. One validation legal win is
handled by the existing rule guard and kept apart, not called a learned
teacher decision. No stage3B student fit has started.

| Intended source route, not prescribed action | TRAIN cases / labels | Validation cases / labels |
|---|---:|---:|
| Kokushi |82 /82|14 /14|
| Chuuren |80 /80|16 /16|
| Suuankou |74 /74|22 /22|
| Daisangen |80 /80|16 /15|
| Chiitoitsu comparison |76 /76|20 /20|
| Unselected random-initial-hand control |28 /28|4 /4|

The five targeted routes each have96 source cases globally, from16 templates,
three damage constructions and two physical-world variants. They are not96
independent family observations. Template damage is not actual shanten or a
guarantee of route viability. The cases deliberately oversample rare shapes;
their frequencies are not natural-game frequencies or a student win rate.

## Actual teacher actions, including refusals

| Action | TRAIN labels | Validation labels |
|---|---:|---:|
| Discard |312|69|
| Riichi |82|18|
| Nine-terminals abort |26|4|

All30 aborts occur in the kokushi-intended source group. They are recorded as
the teacher's actual choices, not replaced with a scripted instruction to
pursue kokushi. This first-turn effect is a reason to add genuine midhand
contexts and inspect route-value labels before a3B recipe; these choices
alone do not establish whether aborting or continuing is optimal. No kan is
selected in this cohort despite some offered ankan menus. The existing
calls/progression replay remains important.

## Construction, separation and limitations

Every source has a complete136-physical-tile wall and an actual native deal
through the existing Mahjong Soul East/red-five engine. The observed initial
14 tiles match the plan. Family components connect matching initial13 and
decision14 patterns under suit permutations/rank reflections. The source
partitions are fixed before native collection; family, canonical shape and
exact public-input overlap across partitions are zero.

Each teacher query uses a fresh Bot/PlayerState and the actual public history
of its own reset environment. Source hints, target route, other concealed hands
and the true wall are never teacher inputs. All512 teacher own-hand history
checks pass;511 mapped network actions are executed natively. Raw walls,
requests, labels and execution evidence remain private. The original active
stage3A source files, data mixture and parameters are not modified.

Generator318642 on ws-l1-002 completes0:0 in3 seconds. Teacher collection318648
on the same GPU node completes0:0 in11 seconds. Both use finite scheduler GPU
allocations, companion CPUs, resource checks and logs, without login compute.

This is source/label production, not student evaluation, rare-yaku mastery or
fit admission. Cases are dealer0/first-turn only; no midhand threat, interacting
route abandonment or complete hand/match outcomes have been collected. Teacher
confidence is unmeasured and action optimality is unproven. Next expand actual
native trajectories/alternatives, keep all failures/aborts and source-family
partitions, normalize/tokenize inputs, and register a separate3B fit/replay
recipe. Do not turn these targeted outcomes into a natural-game strength claim.

[Source plan](../plans/curriculum-yaku-scenarios-20261123.md) ·
[Teacher collection](../plans/curriculum-yaku-teacher-20261123.md) ·
[Existing-data coverage](curriculum-data-catalog-20261123.md)
