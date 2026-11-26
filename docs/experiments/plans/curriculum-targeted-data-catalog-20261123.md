# Targeted simulation and reusable curriculum datasets

2026-11-23. User requirement: generate simulated situations for specific
capabilities and organize the resulting datasets for later classified reuse.
This is a prospective data contract. No new generation, dataset inventory,
training update or evaluation result is claimed by this document.

## Curriculum collections

Maintain one immutable source store and versioned dataset manifests. Collections
are indexed views of source records, not duplicated copies of the same examples.
Reuse existing source files, family assignments and hashes by reference.

| Collection | Purpose and required distinctions |
|---|---|
| Basic | Ordinary discard, useful/dangerous honor handling, riichi accept/decline; distinguish seat wind, round wind, value honors and isolated non-value honors rather than teaching a universal honor-first rule |
| Calls | Chi, pon, ankan, daiminkan, kakan; distinguish offered, teacher intent, executed action, PASS and alternative actions; retain post-meld discards |
| Attack and defense / stage3A | Efficiency, win speed, expected points, opponent threat and switching between attack/defense; retain basic/call replay |
| Yaku and route choice / planned stage3B | Common and rare yaku; recognize completion/tenpai, preserve useful tiles, select or abandon routes; distinguish closed/open restrictions and legal compatible combinations |
| Terminal placement / stage4 | Remaining rounds, dealer, all scores, honba and riichi sticks; third/fourth chasing top two and leading players considering faster finishes |
| Natural-distribution reference | Unselected complete trajectories for measuring regressions and ordinary game strength; report separately from targeted challenge sets |

Overlapping tags do not create independent observations. A single decision may
belong to a call category and a yaku route, but its manifest identity and actual
training exposure must remain explicit. Current frozen stage3A inputs, mixture,
labels and original passing/failing grades are not changed by this plan.

## Generate decision problems, not only successful hands

For each targeted route or operation, cover promising continuation, plausible
but inferior continuation, a competitive alternative, and justified abandonment.
These are coverage intentions, not labels assigned by the scenario author.
Include failed pursuit, opponent wins, draws and unfavorable scoring outcomes.
Never collect only wins or use an increased call/rare-yaku frequency as proof
of better play. Generate and log opportunities as well as actual selections.

For rare hands such as kokushi, chuuren, suuankou and daisangen, begin with
near-complete decisions and then harder earlier route choices. Include public
tile exhaustion, threatened opponents, open/closed changes, and suitable scoring
contexts. Do not force pursuit from an unsuitable ordinary hand. A legal win
detected or automatically executed by the engine is a rules check, not evidence
that the student learned to construct the hand.

Prefer genuine native trajectories or documented constrained dealing followed
by native play. Validate tile conservation, red tiles, meld ownership, turn and
legal actions, riichi/furiten and rule-dependent win legality. Any injected
mid-hand fixture must be separately tagged and its required history validated;
it must not be represented as an ordinary naturally reached game.

Scenario generation and labels are separate. Store actual teacher choices as
teacher supervision, rule-derived facts as rule facts, and simulation-derived
action values as estimates with world count, uncertainty, continuation policy
and objective. A constructed scenario does not establish that one action is
optimal. Hidden state may be used by the simulator but must not leak into the
student or teacher observation, route annotations or explanations.

## Minimum reusable catalog and lineage

Each dataset version must have a manifest with source/version/hash references,
rule and engine versions, generator/config versions, generation seeds, family
assignment policy, record counts and quality status. Each record or sidecar
index must retain:

- Stable record, trajectory/family and scenario-template identifiers, source
  location, and references to the original physical history where available.
- Current authoritative TRAIN/validation/test membership, separately from
  inherited producer split metadata; existing held-out membership is preserved.
- Action menu and chosen action, action category, yaku/route tags, open/closed
  state, progress/difficulty tags and relevant public situation categories.
- Label method/version, teacher identity or simulation objective/budget, and
  confidence/uncertainty where actually measured; unknown fields remain unknown.
- Quality status and reasons for quarantine/exclusion, including legality,
  reconstruction, ambiguity and cross-split duplication findings.

Metadata may live in a sidecar index so immutable original records need not be
rewritten. Labels, outcome tags, source identifiers and future information stay
outside the model-visible input. A route tag is an audit/sampling category, not
an instruction revealing which route the model should choose.

Assign new source families to partitions before collection. All seats, decisions,
alternative rollouts and derived examples from the same trajectory belong to
the same partition. Keep matched templates and close variants together when
forming targeted holdouts, so template memorization cannot impersonate
generalization. Reuse existing whole-family partitions and reserved exclusions;
never recover training data by moving an existing held-out family.

## Training recipes and evaluation

Freeze a separate recipe for each future run: exact dataset manifests, source
and capability weights, replacement/oversampling behavior, epoch order/seed,
exposure budget, held-out sets and capability checks. The catalog itself does
not impose one universal mixture. Publish both original unique family/record
counts and actual weighted exposures; oversampling is not additional evidence.

Keep ordinary/basic/call replay in later courses. Determine rare-yaku weight
after reviewing actual coverage and label quality, before fitting; do not
silently insert it into the already frozen stage3A run. No exact rare-yaku
quota or completed inventory is asserted here.

Use small fixed development suites to diagnose operation/route accept and
decline behavior, preservation of useful tiles, route abandonment and basic
retention. Report each category and its support, not only a macro average that
can hide daiminkan or chi failure. Use separate complete natural-distribution
matches for win/dealin/net/placement and paired uncertainty. Targeted oversampled
win rates are not natural-game win rates. Never select weights or tune mixtures
on the unopened final test; repeated development sets are not final confirmation.

Future actual inventories should report unique trajectories, original records,
label sources, partitions, coverage gaps and exclusions for each collection.
Public outputs remain plans, code and aggregate counts/curves. Raw hands, worlds,
labels and per-case diagnostics retain their existing private scope.

This document only records the requirement. Generation, indexing, deduplication,
validation, tests and fitting must use finite scheduler GPU allocations with
companion CPUs, resource checks, logs and exits. No computation on login nodes,
CPU-only workaround or unmanaged background worker is authorized.
