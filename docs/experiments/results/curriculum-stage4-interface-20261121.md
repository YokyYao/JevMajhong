# Stage4 win/pass interface audit

2026-11-21, Asia/Dubai. This is a native-path audit before admitting terminal-placement data or fitting. It loads no student/v8 weights, changes no current training or practical policy, and produces no optimal-action labels.

The pinned `PolicyAgent.act` returns a legal ron/tsumo immediately, before querying the model. This explains why early-course matches cannot demonstrate learned refusal of an insufficient win: that decision currently bypasses the pointer. The native legal-action representation itself retains the alternative. A ron can be declined through PASS; a tsumo can be declined by a legal discard, rather than an invented pass action.

GPU job311884's allocated companion resources ran `tools/audit_curriculum_win_pass.py` on eight fixed diagnostic seeds2026400000–2026400007, separate from reserved final-test seeds2026300000–2026300031. Four Efficiency agents played each match; at most one ron and one tsumo per match were deliberately declined when offered. Each offer's model-visible request was checked against the exact native legal menu, including hora. A spy client confirmed that the existing policy takes the win without model inference.

| Observation | Count |
|---|---:|
| Complete East matches with internal ledger checks |8|
| Win offers where current policy bypassed inference |41|
| Ron offers / with a native pass alternative |21/21|
| Tsumo offers / with a native discard alternative |20/20|
| Ron declines actually executed |7|
| Tsumo declines actually executed |8|
| Scheduled East4 win offers / currently third or fourth |8/3|

Every deliberately declined branch continued to a completed match without a native-action error, and its internal score ledger reconciled. The eight games contain49 hands. This checks the native decline path and internal accounting; it is not independent han/fu verification or a strength result. Deliberately passing a win is not evidence that passing was correct.

Before stage4 can train, implement and validate a separately selected learned win/decline route, generate representative score-gap/rank states from public information, and label accept/decline using sufficient paired futures and a frozen terminal objective. Account for dealer continuation, East/South extensions, honba, sticks, furiten and alternate ron versus tsumo payments. An East4 observation is not automatically a match-ending state, and a1000-point win is not automatically wrong when behind.

Keep the practical frozen v8 policy's existing win guard in its reference arm. Do not disable its guard and present the resulting behavior loss as evidence that the student surpassed the deployed v8. Any new student routing and its contribution need explicit protocol/reporting alongside the common rules, seeds and seats.

Run the check inside an allocated GPU job after source registration:

```bash
python tools/audit_curriculum_win_pass.py \
  --registered runs/curriculum-optimization-v2/fit-plan/registered.json \
  --out runs/curriculum-optimization-v2/stage4-win-path-audit-private.json
```

The [full curriculum goal](../plans/curriculum-optimization-20261121.md) remains active. The interface audit does not admit stage4 data, start a fit, satisfy the placement objective or replace the unseen final comparison.

## Explicit stage4 model route

`tools/curriculum_endgame.py` now provides a separate `EndgamePolicyAgent`. Only legal-win decision windows in scheduled East4 or later extension rounds consult the supplied model. This is a late-decision scope, not an assertion that East4 must terminate the match. Earlier wins retain the existing immediate-win guard. All non-win windows delegate to the pinned ordinary agent. No practical v8 or active stage1 policy uses this new route.

Within the late window, one model call scores the exact native menu, including hora and ron PASS/tsumo DISCARD. A score-gap threshold does not override its choice. This makes it possible for a later trained model to decline an insufficient win while preserving legal actions. A new route alone does not teach when declining is valuable.

`tools/audit_curriculum_endgame_route.py` used24 distinct diagnostic seeds2026400100–2026400123, separate from the reserved final-test grid. Scripted one-hot selectors exercised both acceptance and decline; these selectors are test fixtures, not a teacher, training label or learned policy. Every game completed and passed its internal settlement ledger.

| Observation | Count |
|---|---:|
| Late ron / tsumo windows routed to one model-interface call |20/17|
| Ron declines / model-interface accepts |13/7|
| Tsumo declines / model-interface accepts |10/7|
| Earlier legal wins retaining the immediate guard |89|
| Late windows from starting rank1/2/3/4 |11/10/9/7|
| Late windows with self as dealer |9|
| Late windows in extension rounds |10|

All37 late windows additionally checked that the unchanged reference agent takes the offered legal win with zero model calls. Public requests contain no wall, opponent hand, future result or label; the scripted selector receives exactly the native public request. The routing check verifies implementation and native continuations, not hidden-world sampling, independent han/fu correctness, a useful learned endgame strategy or v8-level strength. Both types of decline can lead to more hands, so subsequent labels must evaluate the complete match rather than stop at the intervention hand.

The33 curriculum regression checks pass using the repository's discovery entry point, without modifying frozen active-fit files.

```bash
python -m unittest discover -s tests/core -p 'test_curriculum*py'
python tools/audit_curriculum_endgame_route.py \
  --registered runs/curriculum-optimization-v2/fit-plan/registered.json \
  --out runs/curriculum-optimization-v2/stage4-endgame-route-private.json
```

All commands still require allocated GPU resources with companion CPUs. New private window records remain unpublished. Stage4 public-state generation, reliable paired full-match labels, parent admission, fitting and final evaluation remain outstanding.
