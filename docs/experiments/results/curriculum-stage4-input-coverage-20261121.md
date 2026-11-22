# Stage4 source coverage before terminal labelling

Updated2026-11-22, Asia/Dubai. These are unlabelled source inputs, not student predictions, teacher targets or strength evidence. [Recovery362 is complete](curriculum-attack-recovery-final-20261122.md), with a failed strategy-quality grade. The full four-course objective and the untouched final comparison to v8 remain open.

## Registered rule-agent cohort

GPU collection312328 and input audit312330 completed with exit0:0 on a scheduler compute node. Before generation, the collector registered64 fresh logical families2026500000–2026500063 and their whole-family split:52 train/12 validation. Three games ended without a late case; the actual inputs come from50 train/11 validation families. No successful-outcome filtering or early source stop was used. All decisions/seats/hands/score variants of one family share its split.

There are169 distinct decisions:61 progression draws,57 discard-response calls and51 legal-win choices. Replaying each under five score contexts yields845 input rows. The score contexts are first, second, third3,000 behind second, fourth5,000 behind second and fourth13,000 behind second. Deposits/stick banks are conserved. Repeated score views are not independent games.

| Input category | Train rows | Validation rows |
|---|---:|---:|
| Progression / riichi / self-kan menu |250|55|
| Discard-response call menu |235|50|
| Win / decline menu |205|50|
| Total |690|155|

All845 physical action traces reconstruct the exact stored native input/menu. The pinned upstream text encoder accepts every input without truncation or a model backbone load; maxima are949/1,030 packed tokens and512/507 state tokens for train/validation, below4,096 state and8,192 packed limits. The API encoder's unused default label index is not saved as supervision. All169 active recovery launch bindings remain unchanged. The collector's112 curriculum regressions pass.

## Gaps found before fitting

The distinct169 decisions contain153 zero-opponent-riichi and16 one-opponent-riichi cases, with no two/three-opponent-riichi coverage. There are31 chi,29 pon,4 open-kan and4 riichi menus; no self-kan offer appears in this small cohort. First-per-category collection has therefore not met all placement coverage needs.

All41 train win decisions are closed; validation has one open win among ten decisions. An independent public-only solo-win calculation, excluding unknown ura, supports all255 win score views. Only25 views have income at most3,000 points: three train source decisions and two validation source decisions, each repeated across five contexts. There are58 views still outside the top two immediately after the solo transfer. Immediate placement is not terminal placement, particularly for a dealer repeat. Unknown ura and simultaneous wins can change the realised result.

This cohort must not alone establish that the learner sees enough open/small-hand pursuit. Data volume alone would conceal the missing train open-win category. Exact native replays and tokenizer admission are verified; complete ancestral source audit, constructor scope extensions, continuation teacher choice and actual paired terminal labels remain pending.

## Independent Mortal-source supplement

Register a distinct64-family source cohort2026501000–2026501063, with the same predetermined whole-family split and five score contexts. A pinned [Mortal](https://github.com/Equim-chan/Mortal) [298k weight](https://huggingface.co/VoidShine/mortal-298k) acts on all four seats, with legal wins taken for the source-history certificate. This supplements source trajectories; it does not distil its recommended actions as terminal-placement labels.

`tools/curriculum_mortal_source.py` checks the downloaded weight/binary identities, loads weights strictly, rejects exposed opponent hands/draws, mirrors the event stream through a separate native PlayerState and checks its own current hand on every action. Each new game starts new seat states. A missing/unmappable teacher action fails the source job. The adapter is only for consistent newly generated source histories; it does not admit replaying an original private prefix after hidden hands have been sampled.

The first source attempt312338 failed after21 completed games/300 saved score views: the older score-replay helper incorrectly required the live prefix to end at a win trigger and rejected a valid post-call discard decision. The partial files, original source bindings and failed1:0 scheduler exit are preserved. This is an input-preparation failure, not a student training result.

`tools/curriculum_endgame_contexts_v2.py` separates current-hand deposit accounting from the win/furiten history contract and accepts post-chi/pon choices without inventing a draw or win. Six added checks cover live call prefixes, parity with old win scopes, current-hand-only deposits, settled-prefix rejection, invalid deposits/actors and private-field stripping. The original helper and prior successful/failed receipts are unchanged.

Replay312343 completed all64 source games/925 views but its wrapper failed the old partial-row comparison. Structural diagnosis315164 finds295/300 old rows identical in model input, candidate order, metadata and time-ordered physical actions. Five missing IDs are the five score views of one simultaneous-ron window. Native replay verifies both seats0/3 can ron; an unordered source map selected different observing seats. This is a real reproducibility problem in case selection, not an argument for weakening data admission. Both failed attempts remain preserved.

`tools/curriculum_case_order.py` and collector v4 sort simultaneous seats and canonicalize only within-batch actors/consumed sets, preserving physical tiles and time order. GPU315171 completed with exit0:0,123 tests, two full runs of the same64 registered sources and all925 input rows byte-identical. All925 native inputs/menus replay exactly and pass the real tokenizer check. Its independent score audit supports295/310 win views;15 kan-hand views are explicitly outside that scorer's narrow scope. No observed hand/future is supplied to the public score calculation.

The canonical cohort contains185 decisions:63 progression,60 calls and62 wins. There are745 train/180 validation views from51/12 actual families; one registered source ends without a late case. The two repeated runs count as64 families, not128. Distinct source coverage improves as follows:

| Coverage | Rule cohort | Canonical Mortal cohort |
|---|---:|---:|
| Train win decisions |41|50|
| Train open-hand win decisions |0|18|
| Validation open-hand win decisions |1|5|
| Train public solo income≤3,000 decisions |3|8|
| Validation public solo income≤3,000 decisions |2|4|
| Decisions facing two opponent riichi |0|2|
| Decisions facing three opponent riichi |0|0|

The public solo income includes current honba/stick bonuses and assumes one winner/no unknown ura; it is not an intrinsic hand-value class or terminal target. The main training gap in open wins is now represented, but2/3-riichi remains sparse. The next collection improvement is to inspect registered saved physical prefixes for later threat strata that first-per-category caps miss, preserving the existing whole-family split and avoiding duplicate views. Such mining does not generate independent new games.

The source actor checked23,771 own-hand/event alignments and made21,997 network choices, including241 chi,406 pon,15 ankan and8 kakan across the64 full source games. These counts describe source play, not learner skills, opportunities or training targets. Token maxima are1,034/949 packed and610/525 state tokens for train/validation. All169 recovery launch inputs and earlier failed source bindings remain unchanged.

## Threat strata mined from existing prefixes

GPU315188 completed with exit0:0 and126 curriculum checks. A prospective rule scans registered saved physical prefixes in source/time/seat order and retains the first progression/call choice per family and opponent-riichi count1/2/3. It uses no reward or target, does not generate more games and retains the original whole-family split. Prefixes end at previously saved case windows; this is not a complete archive of all later source decisions.

The miner adds96 decisions/480 views:390 train/90 validation, from36/8 existing families. Ten already present score views are skipped. Every new view replays to the exact native input/menu and passes the fixed tokenizer; all169 recovery bindings remain unchanged.

| Added distinct decisions | One opponent riichi | Two opponent riichi | Three opponent riichi |
|---|---:|---:|---:|
| Progression / self-kan choices |44|9|2|
| Call-response choices |35|5|1|
| Total |79|14|3|

The canonical Mortal inputs plus this nonduplicated supplement contain281 decisions/1,405 score views, still from the same64 registered families. Two-riichi windows increase from2 to16; three-riichi from0 to3. Repeated score views and multiple decisions in a family do not increase independent-family counts. Three-riichi remains rare and its conditional world sampler needs explicit efficiency/coverage validation before labels. The added menus include one kakan and one daiminkan offer; absence/frequency of these source opportunities is not a learner performance measurement.

Complete ancestral-source audit, expanded current-world constructors, a validated continuation teacher and actual paired terminal labels remain pending. No stage4 fit, terminal teacher or labels have been admitted; raw hands/worlds/details and weights stay private.

## Data-only reproduction

After creating the stage1 source registration described in the [reproduction guide](../../../reproduction/README.md), submit `cluster/curriculum-stage4-data-v2.slurm` with `JEV_STAGE4_SOURCE=rule` or `mortal`. Its compute, tests, source collection and input audit all run within a bounded GPU job; it never loads a student model or starts fitting. The Mortal route uses canonical collector v4. Output directories must be fresh; an existing or partial cohort is not silently overwritten. Private historical job wrappers are not required by this reusable single-collection/audit template. The two-run reproducibility result above additionally compares an independent repeat with only its output path changed.

The Mortal recipe expects its external dependencies under the configured `teacher_root`: `mortal-298k.pth`, `mortal-upstream/mortal/{model.py,engine.py}` and `mortal-adapter-reference/{_libriichi_loader.py,libriichi/libriichi-3.12-x86_64-unknown-linux-gnu.so}`. The inspected upstream checkout is `0cff2b52982be5b1163aa9a62fb01f03ce91e0d2`. Expected SHA256 values are `bfb3a6c072aa0bfd4171a9cdc77cb6c02ae42cde920843f9e5784394f23447d8` for weights and `3aeb09112198f6babe9bba4dd392ea009669ee7ba5d33cd3cd4242c737bb004b` for the binary. External files are not redistributed. Different compiled binaries require a separately versioned/verified dependency contract rather than bypassing this check. Registration binds the actual external model/engine/loader sources before source play.

To reproduce the extra repeat/mining checks, run the following inside a bounded GPU allocation after the first canonical collection/audit. Create a copy of the canonical JSON config changing only `out` to `runs/curriculum-stage4-placement-mortal-data-v4/repeat-data`; do not edit the registered original config. Collect that copy with `tools/collect_curriculum_mortal_endgame_cases_v4.py`. Then:

```bash
python tools/verify_curriculum_case_repeat.py \
  --parent-root runs/curriculum-stage4-placement-mortal-data-v4 \
  --repeat-root runs/curriculum-stage4-placement-mortal-data-v4/repeat-data \
  --out runs/curriculum-stage4-placement-mortal-data-v4/preparation-execution-private.json
python tools/mine_curriculum_threat_cases.py \
  --parent-root runs/curriculum-stage4-placement-mortal-data-v4 \
  --out runs/curriculum-stage4-threat-prefix-v1/data
python tools/audit_curriculum_endgame_cases.py \
  --data-root runs/curriculum-stage4-threat-prefix-v1/data \
  --out runs/curriculum-stage4-threat-prefix-v1/input-audit-v1
```

The public repeat verifier rejects changed source/split/selection rules and changed dependencies; only the output path may differ. Its receipt supplies the miner's deterministic-parent requirement without historical private job wrappers. All outputs must be fresh; preserve existing or failed receipts. These commands admit neither a continuation teacher nor stage4 optimization.
