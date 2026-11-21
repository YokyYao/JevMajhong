# Stage4 history-consistent current worlds and score-context replay

2026-11-21, Asia/Dubai. This prepares terminal-placement data while risk-recovery312301 is training. No stage4 labels or optimizer updates are admitted. The active fit's169 bound inputs remain unchanged.

## Native reaction-cache issue and correction

RiichiEnv0.4.11's [hand setter](https://github.com/smly/RiichiEnv/blob/cd450aef2cc3d0dc8654bf81f314c0c82f9c9829/riichienv-python/src/env.rs#L153) updates the concealed hands. During a discard response, [legal actions read the existing claim cache](https://github.com/smly/RiichiEnv/blob/cd450aef2cc3d0dc8654bf81f314c0c82f9c9829/riichienv-core/src/state/legal_actions.rs#L252). Replacing hidden hands alone can therefore preserve offers for an old hand and miss a new opponent ron/call. This matters directly to accept/decline counterfactuals.

`tools/curriculum_native_worlds.py` reconstructs ordinary discard reactions in an isolated native replay scratch copy, restores the actual public physical discard identity and transfers the fresh claim cache/active players. It preserves the observer's exact input/menu. The adapter also matches live processing's shape-complete/no-yaku temporary-furiten behavior, which replay event application alone omits.

`tools/curriculum_public_history.py` projects current-hand public events without concealed initial hands or draw identities. It derives non-riichi temporary furiten since the latest own discard. Declared-riichi proposals requiring a past declined opponent ron are rejected under an explicit restricted prior; the generated source agents accept every legal win. Proposed hands and public history are sampler inputs; actual private hands are used only in a separate truth-positive control.

The prior remains conditional on a limited set of public constraints. Unknown private action histories are not fully reconstructed, and this is not a calibrated public-action posterior. Current robbed-kan world construction still needs a separate audit. The adapter checks the exact installed engine version; it does not modify that package or active training files.

## Complete-match continuations

Both branches share their sampled current world. At an already settled hand boundary, the fixture gives the next hand a fresh paired seed while retaining all public scores, dealer, wind, honba and sticks. Its merged financial event stream is reconciled through match termination. This avoids reusing one original episode seed for every world's later hands.

All subsequent actions in this check use public-observation Efficiency agents. These are integration fixtures, not the eventual continuation teacher or training targets. The earlier16 independently checked immediate payments, including freshly sampled ura markers, remain a separate scoped check; this audit does not independently verify han/fu for every complete game.

The source-bound final preparation check ran inside GPU allocation312301 on ws-l2-010. It reused inspected diagnostic logical seeds2026400500–2026400516; repeated runs are reproducibility checks, not independent samples. Final seeds2026300000–2026300031 remain untouched.

| Actual check | Count |
|---|---:|
| Complete source games |17|
| Truth reaction-cache / temporary-furiten controls |68|
| Sampled scheduled-East4/extension win windows |16|
| Starting ranks1 /2 /3 /4 |4 /3 /4 /5|
| Ron / tsumo source windows |9 /7|
| Windows with0 /1 riichi opponents |10 /6|
| Proposed physical allocations examined |66,537|
| Tenpai allocations rejected by the additional history restriction |2|
| Current worlds constructed |32|
| Worlds with changed reaction caches |10|
| Accept/decline paired complete matches |32 pairs /64 matches|
| Later-hand reseeds with public context preserved |29|

This sample did not cover two/three riichi opponents. Uniform joint rejection must be profiled on those cohorts before data admission; exhaustion cannot be replaced by an invalid hand or unrecorded biased sampler.

A controlled custom physical wall additionally produces a native no-yaku temporary-furiten flag, then proposes a different opponent hand that can ron. The adapter reconstructs the old flag, clears it for the unrelated proposed hand, exposes and executes the previously absent ron, preserves the observer's input and reconciles payment. The natural sampled cases happened to have zero derived-true temporary flags, so this positive control is essential.

## Same-hand score augmentation

`tools/curriculum_endgame_contexts.py` replays the same per-hand wall and legal action prefix with new score tables. It reverses the prefix's accepted riichi deposits to obtain valid starting scores, retaining the current stick bank. Every context of a source wall must share one training/validation group.

One inspected source hand was reproduced under five contexts: holding first, holding second, third with a3,000-point gap to second, and fourth with5,000/13,000-point gaps. All hand information and legal menus were identical apart from scores; ten accept/decline complete-match fixtures reconciled. This is five views of one source hand. It is not five independent examples of an optimal action.

`tools/curriculum_endgame_ledger.py` supports declared initial score/stick tables, including nonuniform scores and a carried pool. It checks hand handovers, deposits, win/draw transfers, terminal tied-leader pool assignment and total100,000-point conservation. It preserves the original64 fixture outcomes and enables the ten augmented-context fixtures. Independent han/fu verification remains separate.

All107 curriculum regressions pass. The final check bound the new source and installed native binary before running, reproduced the complete-game outcomes and score-context replays, and verified all169 active recovery inputs unchanged. Raw case scores, hands, worlds and paired details remain private; this report publishes aggregates and source.

```bash
python -m unittest discover -s tests/core -p 'test_curriculum*.py'
python tools/audit_curriculum_history_worlds.py \
  --registered runs/curriculum-optimization-v2/fit-plan/registered.json \
  --out runs/stage4-check/history-worlds-private.json \
  --seed-start 2026400500 --games 24 --cases 16 --worlds 2 \
  --late-only --full-continuations
```

All commands require bounded scheduler GPU allocations, including companion CPU work, and fresh output paths. Next collect fresh grouped score-augmented draw/call/win cases, validate a public-information continuation teacher, and generate reliable paired terminal targets as described in the [placement-data plan](../plans/curriculum-stage4-placement-20261121.md).
