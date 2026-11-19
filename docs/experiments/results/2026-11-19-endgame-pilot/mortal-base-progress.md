# Fresh-base Mortal distillation: development progress

Status: training is active. This is a new rank-16 LoRA and pointer head on pinned Qwen3.5-0.8B, with no Mahjong-trained v8/audited176 adapter or head and no v8 replay targets. The backbone is frozen. See the [registered plan](../../plans/mortal-base-training-20261120.md).

GPU job309978 passed preflight and seven checks and encoded all22,321 teacher training decisions. Saved-model validation is complete through update640; both LoRA and head changed. The512 validation decisions come from128 whole source games separate from the384 training games. This is an inspected development set, not an untouched confirmation test. This pure-teacher phase covers East1–3 non-win menus under the existing serving guard; East4 pursuit and win acceptance receive no direct teacher supervision here, so agreement cannot establish those abilities.

| Saved step | Weighted teacher CE ↓ | Weighted agreement | Overall agreement, selected512 menus | Unique training decisions exposed |
|---|---:|---:|---:|---:|
| 0 | 2.86008 | 32.22% | 38.67% | 0 |
| 64 | 1.42608 | 57.67% | 51.37% | 3,072 |
| 128 | 1.13246 | 61.30% | 54.49% | 6,144 |
| 192 | 1.02006 | 64.55% | 59.77% | 9,033 |
| 256 | 0.89589 | 67.59% | 63.67% | 11,467 |
| 320 | 0.86502 | 67.03% | 64.84% | 13,899 |
| 384 | 0.83625 | 70.79% | 68.36% | 16,095 |
| 448 | 0.84677 | 69.07% | 65.82% | 18,015 |
| 512 | 0.84989 | 67.40% | 65.62% | 19,935 |
| 576 | 0.81667 | 69.30% | 67.58% | 21,562 |
| 640 | 0.82420 | 70.70% | 69.73% | 21,946 |

Step0 is a random decision head on a pretrained backbone, not an old Mahjong policy or a uniform-random action baseline. Agreement depends on this selected/weighted validation mix and is not a win rate.

| Saved step | Exact teacher-positive chi/pon matches /64 | Exact teacher-negative chi/pon matches /64 | Exact teacher-positive riichi matches /32 | Exact teacher-negative riichi matches /32 |
|---|---:|---:|---:|---:|
| 0 | 63 | 0 | 7 | 9 |
| 64 | 0 | 64 | 30 | 3 |
| 128 | 0 | 64 | 26 | 11 |
| 192 | 16 | 64 | 23 | 15 |
| 256 | 22 | 64 | 29 | 11 |
| 320 | 34 | 62 | 20 | 14 |
| 384 | 37 | 62 | 29 | 13 |
| 448 | 28 | 62 | 27 | 16 |
| 512 | 38 | 62 | 31 | 7 |
| 576 | 38 | 63 | 23 | 15 |
| 640 | 43 | 63 | 28 | 13 |

The initial learner changed from calling too often to always declining chi/pon at64–128, then began selecting some teacher-positive calls at192. Exact positive-call matches change16/64 →22/64 →34/64 →37/64 →28/64 →38/64 →38/64 →43/64 at192/256/320/384/448/512/576/640. Negative-call matches are63/64 at576/640. Weighted CE reaches its current minimum at576, then rises slightly at640 despite improved agreement. No claim of solved decisions or improved playing strength follows from these auxiliary metrics.

| Available decision | Validation opportunities | Teacher selects | Step576 selects | Step640 selects |
|---|---:|---:|---:|---:|
| Riichi | 64 | 32 | 33 | 45 |
| Chi | 75 | 22 | 10 | 12 |
| Pon | 76 | 43 | 30 | 33 |
| Ankan | 26 | 6 | 1 | 5 |
| Daiminkan | 21 | 0 | 0 | 0 |
| Kakan | 17 | 3 | 2 | 4 |

Opportunity rows overlap: a menu can contain several action types. At640,12/12 selected chi and32/33 selected pon match the teacher's action type; exact candidate agreement is43/64 on the teacher-positive call stratum. These are teacher matches, not independent expert correctness labels.

Riichi at640 has28 teacher-positive matches,17 extra declarations,4 misses and15 correct declines. Ankan has3 matches,2 extras and3 misses; kakan has2 matches,2 extras and1 miss. Kan-positive recall improves to5/9 from1/9 at576, with4 extra kan selections relative to the teacher. These remain small, fluctuating counts; neither CE nor action frequency alone establishes playing strength.

## Rare positive-kan exposure

A read-only coefficient/exposure audit at saved384 found20 ankan-positive and10 kakan-positive training records among257 kan-opportunity records, and no daiminkan-positive records. All30 distinct positives had been exposed. The registered coefficient mass on those examples is0.5837% of the total objective in expectation; the actual first384-update schedule assigns0.5859%, with60 ankan-positive and30 kakan-positive draws. Only82/384 updates contain any teacher-positive kan.

These are loss coefficients assigned to positive examples before2% label smoothing, not gradient-norm shares, target-class probability mass or causal proof of the kan failure. Sparse positive exposure is a concrete risk to investigate after the frozen observation period. Rebalancing select/decline exposure could reduce sampling variance while preserving calibrated loss mass; it must be tested rather than assumed to fix the issue. No active weights, data or recipe were changed by this audit.

## Training fit versus held-out kan behavior

A bounded normal-loader diagnostic evaluated all257 kan TRAIN menus at saved384 and448; the existing64 kan validation predictions were reused. It performed514 training-menu inferences and no additional validation inference. Only fresh-base checkpoints were loaded, with no training changes.

| Step | Split | Ankan positive matches | Extra ankan | Kakan positive matches | Extra kakan |
|---|---|---:|---:|---:|---:|
| 384 | TRAIN | 0/20 | 1 | 0/10 | 1 |
| 384 | VALIDATION | 0/6 | 0 | 0/3 | 1 |
| 448 | TRAIN | 11/20 | 12 | 2/10 | 7 |
| 448 | VALIDATION | 2/6 | 2 | 1/3 | 1 |

The384 model fails to recover even the kan positives it has trained on;448 partially recovers them with many extra selections. This is not solely a gap between well-fitted training examples and unseen validation examples. The small positive sets and coefficient audit do not isolate weighting, optimization, capacity or label quality as the sole cause, or prove the teacher's choices optimal. Detailed predictions remain private.

## Early complete-match development diagnostic

After observing512 and before new candidate outcomes, checkpoint384 was chosen for an additional complete-East diagnostic because it had the lowest observed teacher CE at that time. This is a post-hoc development analysis of an underexposed checkpoint, outside the registered post-768 selection rule; the active fit and its selection rule remain unchanged. The reusable evaluator is [tools/evaluate_mortal_checkpoint.py](../../../../tools/evaluate_mortal_checkpoint.py).

The diagnostic completed64 new candidate games and reused64 audited176-reference games on the same eight seeds, four controlled seats and two opponent profiles. It uses the same formatter/guard and normal BF16, unmerged, temperature1 policy. Runtime/source hashes, reference identity and candidate immutability passed; independent accounting/statistics checks covered all128 complete logs. These check payments, pools and ranks, not an independent han/fu scorer.

| Complete-East metric | Audited176 reference | Fresh-base step384 |
|---|---:|---:|
| Games | 64 | 64 |
| Top-two finishes | 31/64 (48.44%) | 25/64 (39.06%) |
| First place | 17.19% | 15.62% |
| Fourth place | 25.00% | 32.81% |
| Mean rank | 2.59375 | 2.78125 |
| Mean net points/game | -1,445.31 | -3,814.06 |
| Hand win rate | 20.63% | 15.30% |
| Deal-in hand rate | 11.17% | 11.90% |

The paired top-two difference is -9.375 percentage points, with whole-seed95% t interval [-20.19,+1.44] percentage points across8 seed groups. Net points are lower by2,368.75/game. This adverse point estimate does not demonstrate improvement; the interval still crosses0 and the small inspected development grid does not establish definitive degradation or general online win rates.

Actual actions over64 games (reference → new): chi60 →9, pon89 →37, ankan9 →2, kakan4 →0, daiminkan21 →0, riichi82 →91. The reference therefore already executes calls in this native environment. Counts alone do not establish which decisions cause the score gap. The new model's lower hand-win rate and sparse calls warrant further training/diagnosis, without claiming a single cause.

This result applies only to step384, which had seen16,095/22,321 unique examples. Step576 later becomes the lowest-CE checkpoint but has not received this complete-game comparison. This diagnostic is not independent confirmation, a causal test of initialization alone, or a model release. Raw logs and research checkpoints remain private. The source code completed this full evaluation with exit0; the main training continues.

## Teacher daiminkan bridge diagnostic

The source pool has no positive daiminkan labels. To check the interface, a bounded diagnostic inside the existing GPU allocation replayed4 complete stored training-source games. All4 inspected controlled-player E1–3 daiminkan windows exposed the kan action in the teacher's real legal mask, but the teacher's natural Q ranking preferred another action. Two separate Bot copies were forced to select kan solely to test decoding: each produced daiminkan and mapped uniquely into the native legal menu. The real source trajectories remained identical to the stored logs; no forced action was applied to a source game or added to training.

This limited diagnostic checks visibility and decoding in those windows, not every source window and not whether the teacher's preferences maximize Mahjong results. The absence of positive labels remains a training-data limitation; matching declines cannot establish learned acceptance of daiminkan. Detailed source events/Q values remain private.

## Schedule and integrity

At640, only21,946/22,321 unique training decisions have been exposed. Continue the fixed schedule: saved validation every64 updates, at least768 updates and full exposure before early stopping, maximum1536. The call-negative population alone needs703 updates at six examples/update for first exposure. No early checkpoint is selected as final before the observation floor. Step576 has the lowest CE through640; the registered selection policy waits for full exposure, which is not a guarantee of the best playing strength.

A prospective full-exposure development screen uses checkpoint768. This was added after the adverse384 result and before observing768 outcomes. If the monotonic registered selector later moves to a different, lower-CE checkpoint, separately evaluate768 on64 games; otherwise its ordinary post-fit comparison supplies the screen. This avoids repeating an identical final candidate. Reuse the64 reference games and measured384 results. Compare top-two and mean net points together to guide continued observation versus a revised recipe; a positive95% bound is not required merely to continue research, and a favorable point estimate is not a qualified improvement. Active optimizer/data/selection code remain unchanged; any externally triggered stop or new recipe must document its checkpoint and cleanup.

A read-only companion-CPU step inside the same GPU allocation verified all10 base snapshot files (1,769,905,646 bytes) against their original content addresses, including weights and tokenizer. Nested links to shared storage are handled using the snapshot's content address. This is local pinned-cache integrity, not a new remote signature. Registered training sources/data match their frozen hashes, and eight published source/plan/report files match their Git blob identities. Post-fit checks were compiled and frozen; they repeat base integrity checking after fitting.

A separate bounded companion-CPU check verified current RiichiEnv0.4.11 Python/native files against its installed wheel RECORD and replayed all64 cached reference-model games using their recorded actions. Every native event, terminal score and controlled-seat rank was reproduced exactly. This took9.83 seconds, loaded no baseline policy and generated no new baseline-policy games. Preserve the recorded native-file hashes through final comparison; this is additional evidence for cache reuse, not an independent test of alternative trajectories or Mahjong scoring rules. Raw replay details stay private.

An input-isolation check also processed all22,833 training/validation records: public state/instruction/option serialization matched the normal inference path and stayed identical after mutating labels, target distributions, source names and metadata. Fourteen representatives (two per stratum) had identical actual forward-input token arrays across training, inference and mutated-supervision forms, with labels genuinely changed. This excludes those supervision fields as a visible-input shortcut; it does not establish every feature's sufficiency or playing strength. No policy weights were loaded, and registered training code/data stayed unchanged.

After fitting, reload/audit selected and final checkpoints and evaluate the registered selected candidate against the same64 reference games. The optional768 screen above is only separate once768 can no longer be that selected candidate; no reference-policy rerun is needed. Final-candidate results remain pending; early384 is a separate experiment. Dependency-bound GPU audit job309982 follows a successful fit and is canceled automatically if that dependency fails. Training code8610bff1 passed CI. The reusable evaluator and adverse384 result were published in987ee146; its core and desktop CI jobs44637163535 both passed. Raw records, prediction details and research weights remain private.
