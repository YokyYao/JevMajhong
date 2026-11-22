# Live worlds and conditional terminal-value scout

2026-11-22. The fresh-base four-course goal remains active and unachieved. No new student fit or stage4 target is admitted by this report. The completed risk recovery retains its failed quality grade and gap to frozen v8. The final32-seed complete-East comparison is unopened. Practical v8, main and Windows releases are unchanged.

## Native environment coverage

GPU315334 completed with exit0:0 and140 curriculum checks. Sixteen prospectively selected scope/category/threat buckets cover own draws, discard responses, own post-chi/pon discards and legal-win choices, including0/1/2/3 opponent riichi. The matrix uses nine source families,32 hypothetical public-prior worlds and64 full-terminal arms. All original-world menu/cache/furiten controls pass; four sampled reaction caches differ from their source, so copying the source cache would have been wrong. All arms settle and their financial ledgers reconcile;25 later-hand resets use independent paired seeds while preserving public match context.

`curriculum_hierarchical_prior.py` declares a new conditional prior: randomize riichi-seat order, sequentially draw uniform physical concealed hands subject to native tenpai, then draw other hands/wall and reject incompatible public missed-win histories. It matches the direct uniform-joint conditional distribution for0/1 riichi, **not** for2/3. Equal weights refer only to this declared hierarchical prior; there is no importance correction, calibrated action-history posterior or hidden-state likelihood claim. The matrix consumes134,683 native hand draws with one history rejection and no exhausted bucket. The three-riichi draw/response fixtures complete in about0.20/0.32 seconds for two worlds on their assigned node. This is sampler feasibility, not a strength result.

The constructor keeps the observer's actual concealed tiles, public input and legal menu; it repartitions only the unknown inventory, refreshes discard reactions and derives sampled opponents' temporary flags. Sampled opponents retain the restricted no-prior-declined-riichi-ron contract. Current robbed-kan decision windows remain outside this adapter; earlier kan events are retained. The internal ledger is not independent han/fu verification.

Actual known-observer decline fixtures subsequently pass inside GPU315393: a legal ron is passed, another same-cycle ron is blocked, and true-world reconstruction retains the input/menu. After the observer's next draw/discard, non-riichi ron becomes available again while riichi ron remains blocked. Both original and reconstructed native states agree. An earlier fixture315380 failed because exact physical-copy selection ignored native candidate deduplication; the original code/failure remain private and unchanged. Passing v2 compares the legal non-red tile type. These two fixtures do not prove every possible nested sampled-history construction.

## Frozen-v8 continuation and four hard win choices

The public-observation continuation pilot315344 executes16 complete arms and261 real model calls, with no model update. Its rollout takes21.79 seconds and initial loading41.44 seconds on that assigned node. The controlled seat uses unchanged frozen v8, the next seat a calling opponent and the remaining seats efficiency opponents. Later legal wins use the standard forced-win guard. This is a conventional continuation, not an optimal multi-decline pursuit teacher; it can undervalue repeated pursuit.

The [32-world protocol](../plans/curriculum-terminal-scout-20261122.md) selects four TRAIN-only, nondealer legal wins using public rank, win kind, independently calculated no-ura solo income≤3,000 and rank still outside the top two after accepting. Selection uses no simulated outcome. Scout315376 completes256 terminal arms with exit0:0 and143 checks. Its preceding submission315375 stops before model loading because its predecessor has been purged from live scheduler state; successful predecessor execution/exit receipts remain available, while the scheduler history database is unavailable. This missing historical query is recorded rather than invented as a terminal scheduler result.

The [fixed follow-up](../plans/curriculum-terminal-scout-extension-20261122.md) extends **all four** cases to128 worlds with the original seeds/arms and the same policy/prior. It reuses the original32 paired worlds and computes768 additional arms. All1,024 total arms complete; the new rollouts take178.62 seconds and3,859 model calls, plus7.87 seconds for loading. Job315380 then fails in the unrelated physical-copy fixture described above: preserve its FAILED exit, and distinguish its completed scout artifact from full wrapper success. The corrected native decline fixtures run successfully afterward in315393.

| Fixed starting situation | Accept top-two /128 | Decline top-two /128 | Accept mean net change | Decline mean net change |
|---|---:|---:|---:|---:|
| Third,1,000-point ron |0|82|+1,000.00|+1,969.53|
| Third,2,500-point tsumo income |0|53|+2,500.00|+629.69|
| Fourth,5,000 behind second,1,000-point ron |0|1|+1,000.00|+1,005.47|
| Fourth,13,000 behind second,3,000-point ron |0|10|+3,000.00|+2,665.63|

These are four conditional position values, **not the student's match top-two rate**. Tsumo decline is one fixed legal discard, not an optimized best discard. The second/fourth rows illustrate that greater top-two probability may cost expected points. The5,000-gap row remains sparse even at128 worlds and should not become a confident hard target. None of these cases admits a universal "never accept a small win when behind" rule. Other cases can favor accepting, especially a sufficient transfer or dealer continuation.

The paired top-two difference (accept minus decline) is−64.06,−41.41,−0.78 and−7.81 percentage points. The approximate Wilson/Bonferroni paired-discordance intervals are[−72.87,−50.41],[−51.31,−28.38],[−5.17,+3.66] and[−14.86,−0.18] points. Bonferroni protection here is for the two marginal discordance probabilities inside one interval; these are descriptive pointwise intervals, not familywise guarantees across four cases, repeated prefixes, policy choice or prior misspecification. All-zero32-world outcomes explicitly retained nonzero-width uncertainty;128 was the fixed maximum, not a significance stopping rule.

Raw hands, full trajectories, world allocations and per-record results stay private. No old failure or parent-admission gate changes. A future label plan must explicitly choose conventional one-step policy improvement or a separately verified pursuit continuation, account for sparse tails, mask unobserved alternatives, verify serving representation and source exclusions, and register its parent/mix/budget before fitting.

## Basic-course repair is separate

The first-course riichi problem remains a priority. The [new data plan](../plans/curriculum-basic-teacher-data-20261122.md) registers256 independent true-trajectory Mortal families,205 TRAIN/51 validation, before generation. It collects all eligible closed-hand riichi choices plus bounded early ordinary-discard slots for each seat. It does not initialize a student from Mortal, reuse a private prefix after resampling, label invented score contexts with stale source actions, or add a rare-yaku bonus. Source-play yaku counts are diagnostics, not yaku mastery. This collection is running in bounded GPU315393 at report preparation, with146 checks and the known-decline fixtures passed. Its dependent GPU audit315396 will reconstruct/tokenize/normalize the complete cohort only after successful collection; no fit starts automatically.

## Running the exploratory tools

These commands require the registered source/runtime artifacts and the canonical925+480 input preparation in the [input coverage guide](curriculum-stage4-input-coverage-20261121.md). Run inside a bounded GPU allocation, with fresh outputs. The root paths are explicit research prerequisites, not a downloadable private dataset or a claim that missing historical artifacts are public.

```bash
python tools/audit_curriculum_live_worlds.py \
  --out runs/curriculum-live-worlds-v2/matrix-audit-v1
python tools/benchmark_curriculum_continuation.py \
  --out runs/curriculum-live-worlds-v2/v8-continuation-pilot-v1
python tools/scout_curriculum_terminal_values.py \
  --out runs/curriculum-terminal-scout-v1/scout-v1
python tools/extend_curriculum_terminal_scout.py \
  --parent runs/curriculum-terminal-scout-v1/scout-v1 \
  --out runs/curriculum-terminal-scout-v1/scout128-v1
python -c 'from tools.audit_curriculum_known_declines_v2 import audit; print(audit())'
```

Basic data collection uses `tools/collect_curriculum_basic_teacher.py --config configs/training/curriculum-basic-mortal-data-v1.json`; the separate `tools/audit_curriculum_basic_teacher.py` creates normalized private training inputs after full native/source checks. Never run these computations on a login node. Actual source/label audits and the eventual model-quality/final comparison are distinct requirements.
