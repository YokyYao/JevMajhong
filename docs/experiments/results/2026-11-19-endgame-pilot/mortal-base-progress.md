# Fresh-base Mortal distillation: development progress

Status: training is active. This is a new rank-16 LoRA and pointer head on pinned Qwen3.5-0.8B, with no Mahjong-trained v8/audited176 adapter or head and no v8 replay targets. The backbone is frozen. See the [registered plan](../../plans/mortal-base-training-20261120.md).

GPU job309978 passed preflight and seven checks and encoded all22,321 teacher training decisions. Saved-model validation is complete through update1088, with the observation floor and full exposure satisfied; both LoRA and head changed. Step1088 is now the current candidate, with the lowest CE so far. The512 validation decisions come from128 whole source games separate from the384 training games. This is an inspected development set, not an untouched confirmation test. This pure-teacher phase covers East1–3 non-win menus under the existing serving guard; East4 pursuit and win acceptance receive no direct teacher supervision here, so agreement cannot establish those abilities.

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
| 704 | 0.83233 | 71.24% | 69.92% | 22,321 |
| 768 | 0.80443 | 71.14% | 70.12% | 22,321 |
| 832 | 0.77506 | 74.49% | 73.24% | 22,321 |
| 896 | 0.78599 | 72.65% | 71.68% | 22,321 |
| 960 | 0.79435 | 70.57% | 69.53% | 22,321 |
| 1024 | 0.77999 | 72.25% | 70.70% | 22,321 |
| 1088 | 0.76746 | 70.61% | 69.14% | 22,321 |

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
| 704 | 42 | 63 | 27 | 16 |
| 768 | 42 | 62 | 25 | 13 |
| 832 | 44 | 62 | 27 | 17 |
| 896 | 45 | 63 | 27 | 14 |
| 960 | 41 | 62 | 22 | 15 |
| 1024 | 40 | 63 | 25 | 13 |
| 1088 | 39 | 63 | 30 | 13 |

The initial learner changed from calling too often to always declining chi/pon at64–128, then began selecting some teacher-positive calls at192. Exact positive-call matches reach44/64 at832, with62/64 negative-call matches. CE reaches a new low at832, but the per-action results remain mixed. No claim of solved decisions or improved playing strength follows from these auxiliary metrics.

| Available decision | Validation opportunities | Teacher selects | Step704 selects | Step768 selects |
|---|---:|---:|---:|---:|
| Riichi | 64 | 32 | 37 | 38 |
| Chi | 75 | 22 | 11 | 12 |
| Pon | 76 | 43 | 33 | 33 |
| Ankan | 26 | 6 | 1 | 6 |
| Daiminkan | 21 | 0 | 0 | 0 |
| Kakan | 17 | 3 | 2 | 4 |

Opportunity rows overlap: a menu can contain several action types. At768,12/12 selected chi and31/33 selected pon match the teacher's action type; exact candidate agreement is42/64 on the teacher-positive call stratum. These are teacher matches, not independent expert correctness labels.

Riichi at768 has25 teacher-positive matches,13 extra declarations,7 misses and19 correct declines. Ankan has3 matches,3 extras and3 misses; kakan has2 matches,2 extras and1 miss. Kan action-type positive recall returns to5/9, with5 extra selections relative to the teacher. These remain small, fluctuating counts; neither CE nor action frequency alone establishes playing strength.

At832, riichi has27 positive matches and10 extra declarations; chi has12 positive matches with no extras, and pon has33 matches with2 extras. Ankan falls to2/6 positive matches with1 extra; kakan has1/3 with1 extra. The lower overall CE therefore coexists with weaker positive-kan recall than768. Daiminkan still has no positive teacher labels or selections.

At896, weighted CE increases to0.78599, so832 stays selected with one stale check. Exact positive/negative call matches improve to45/64 and63/64, while riichi has27 positive matches with13 extras. Ankan matches1/6 with1 extra and kakan1/3 with1 extra. Continue the registered observation rule; this single fluctuation does not trigger a restart or establish improving strength.

At960, weighted CE is0.79435 and832 remains selected with two stale checks. Exact positive/negative call matches are41/64 and62/64. Riichi has22 positive action-type matches with12 extras; ankan2/6 with1 extra and kakan1/3 with1 extra. The fit remains active under the four-stale-check rule. If neither1024 nor1088 improves the registered CE threshold, it will stop at1088 and compare the retained832 candidate; a later improvement changes that conditional stopping point. No960 complete-match score has been measured.

At1024, CE recovers to0.77999 but remains above832's0.77506; there are now three stale checks. Exact positive/negative call matches are40/64 and63/64, riichi25/32 and13/32. Ankan has2/6 action-type positives with1 extra, and kakan1/3 with1 extra. If1088 does not improve the registered threshold, fitting stops there; otherwise selection/observation continues. No1024 complete-match score has been measured.

At1088, CE improves to0.76746, replacing832 as the current selection and resetting stale checks to0. The conditional1088 stopping point therefore does not apply. Exact positive/negative calls are39/64 and63/64, riichi30/32 and13/32; the learner declares riichi45 times versus32 teacher selections, including15 extras. Ankan has2/6 action-type positives with1 extra and kakan1/3 with1 extra. Overall agreement declines while weighted CE improves; these mixed auxiliary metrics do not establish stronger play. No1088 complete-match score has been measured. Continue the registered recipe, with the1536 cap unchanged.

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

## Full-exposure768 complete-match screen

The prospectively registered768 screen completed64 new candidate games after the monotonic teacher-CE selector moved to832. It reused the same64 reference games and performed no new reference-policy inference. All128 complete logs passed accounting/statistics checks, with evaluator/runtime hashes and candidate identity unchanged. The bounded GPU step exited0. No384 games were repeated.

| Complete-East metric | Audited176 reference | Fresh-base step384, already measured | Fresh-base step768 |
|---|---:|---:|---:|
| Games per policy | 64 | 64 | 64 |
| Top-two finishes | 31/64 (48.44%) | 25/64 (39.06%) | 28/64 (43.75%) |
| First place | 17.19% | 15.62% | 21.88% |
| Fourth place | 25.00% | 32.81% | 31.25% |
| Mean rank | 2.59375 | 2.78125 | 2.65625 |
| Mean net points/game | -1,445.31 | -3,814.06 | -1,435.94 |
| Hand win rate | 20.63% | 15.30% | 18.72% |
| Deal-in hand rate | 11.17% | 11.90% | 11.73% |

For768 versus the reference, top-two difference is -4.6875pp, with whole-seed95% t interval [-24.78,+15.40]pp across eight groups. Net points differ by only +9.375/game. These are not evidence of a reliable improvement. Compared descriptively with the already measured384 checkpoint on the same development grid,768 gains4.6875pp top-two and2,378.125 points/game; no independent interval or confirmation claim is attached to that across-checkpoint comparison.

Over64 games,768 selected chi17, pon51, ankan6, kakan2, daiminkan0 and riichi98; the reference counts are60/89/9/4/21/82 respectively. Calls remain sparse and hand-win rate lower than the reference. These totals are not opportunity-adjusted and do not isolate a cause of the performance gap. More training improved the observed384 point estimates but has not established superiority. Continue the frozen training/teacher-CE selection schedule and compare its later final selection; do not deploy or revise the active loss from this single small screen. E4 pursuit and win acceptance still lack direct teacher supervision in this phase.

This is an inspected development comparison using reused seeds/reference games, not independent confirmation or an online win-rate estimate. Detailed logs and research weights remain private; payment auditing is not an independent han/fu scorer.

## Teacher daiminkan bridge diagnostic

The source pool has no positive daiminkan labels. To check the interface, a bounded diagnostic inside the existing GPU allocation replayed4 complete stored training-source games. All4 inspected controlled-player E1–3 daiminkan windows exposed the kan action in the teacher's real legal mask, but the teacher's natural Q ranking preferred another action. Two separate Bot copies were forced to select kan solely to test decoding: each produced daiminkan and mapped uniquely into the native legal menu. The real source trajectories remained identical to the stored logs; no forced action was applied to a source game or added to training.

This limited diagnostic checks visibility and decoding in those windows, not every source window and not whether the teacher's preferences maximize Mahjong results. The absence of positive labels remains a training-data limitation; matching declines cannot establish learned acceptance of daiminkan. Detailed source events/Q values remain private.

## Schedule and integrity

### Endgame data retained for possible follow-up, not yet trained here

A bounded read-only preparation audit checked the existing48 terminal-outcome training positions and8 separate monitor positions:434 legal actions with all64 paired outcomes each (27,776 total). Label hashes, manifest membership, world designs, stored completion/ledger receipts and terminal score-to-rank consistency match. Source games are disjoint from the512 teacher sources and the8 full-match comparison seeds. This is receipt/schema consistency checking, not new native resimulation or independent scoring.

Their existing targets mix an old128-step policy prior with simulated top-two outcomes and cannot be presented as Mortal labels. A diagnostic constructed hypothetical uniform-anchor targets from terminal outcomes without reading that old prior or existing soft targets. The outcomes themselves still depend on two old continuation policies (formal v8 and parent128), and fixed current opponent hands; removing the explicit prior does not remove that behavioral dependence or create a public-state posterior. No targets have been admitted into this fresh-base fit, and no old model was loaded. A later retention stage needs an explicit recipe, validation and these limitations disclosed; this audit alone does not restore pursuit ability.

Checked upstream `libriichi/src/state/update.rs` sets `is_all_last` false in East and true at South4; its observation encoder also distinguishes round wind and hand index. This source inspection is not a probe of the loaded extension or proof of the teacher's endgame preferences. It is a reason to verify East-only endgame alignment before treating raw teacher choices as pursuit truth. The current E1–3/no-win foundation recipe remains unchanged.

A separate probe then loaded the actual pinned native extension, without policy weights. For one public East4 initial state, adding `game_mode`/`game_length` fields requesting East-only versus half-game length to `start_game` produced identical1012×34 observation features and46-value action masks. Changing its wind from East to South changed the features. This only verifies those two proposed input fields and that single state; it does not expose the internal all-last flag, evaluate teacher Q preferences, or justify changing wind in training labels. Opponent concealed tiles were masked and active data/weights stayed unchanged.

An explicit [follow-up retention plan](../../plans/mortal-base-retention-20261120.md) and separate preparation/training tools now address this missing scope. They accept only the fully audited fresh foundation, recompute targets with its own prior, and mix terminal supervision15% with teacher replay85%. New source-isolation, incomplete-world rejection and old-target-independence tests pass on the allocated GPU node. Source commit `d5b4087c8cf002450c93ced19423841abab2b55f` passed both core and desktop CI44647125471; the five published code/plan blobs match local bytes, and private audit/Slurm scripts were compiled and frozen. Preparation310238 depends on successful foundation audit309982, fit310240 on preparation, and audit310241 on fitting, with invalid dependencies canceled automatically. All are still pending dependencies; no retention training has started. The current foundation source/data/selection are unchanged.

At704, all22,321 unique training decisions have been exposed. A bounded read-only audit loaded the saved704 trainer state: its seen-ID set exactly equals the training set, contains none of the512 selected validation IDs, and all376 stored optimizer parameter states report704 updates. Registered code/data hashes still match. This is direct coverage/update evidence, not a claim of competence or source-game independence; the whole-game split was separately verified during preparation.

The768 observation floor is now satisfied. Continue the registered validation every64 updates, with up to1536 updates and four stale checks after the floor/full exposure. Step1088 replaces832 as the lowest-CE eligible selection; the selection can change. Complete exposure does not guarantee playing strength.

A prospective full-exposure development screen uses checkpoint768. This was added after the adverse384 result and before observing768 outcomes. If the monotonic registered selector later moves to a different, lower-CE checkpoint, separately evaluate768 on64 games; otherwise its ordinary post-fit comparison supplies the screen. This avoids repeating an identical final candidate. Reuse the64 reference games and measured384 results. Compare top-two and mean net points together to guide continued observation versus a revised recipe; a positive95% bound is not required merely to continue research, and a favorable point estimate is not a qualified improvement. Active optimizer/data/selection code remain unchanged; any externally triggered stop or new recipe must document its checkpoint and cleanup.

The selector moved to832, so the one-time768 screen ran in a bounded step of the existing GPU allocation and completed with exit0. Its complete results are above. No reference policy was reloaded or384 games repeated; partial-game outcomes were not used for a strength claim or adaptive stop.

A read-only companion-CPU step inside the same GPU allocation verified all10 base snapshot files (1,769,905,646 bytes) against their original content addresses, including weights and tokenizer. Nested links to shared storage are handled using the snapshot's content address. This is local pinned-cache integrity, not a new remote signature. Registered training sources/data match their frozen hashes, and eight published source/plan/report files match their Git blob identities. Post-fit checks were compiled and frozen; they repeat base integrity checking after fitting.

A separate bounded companion-CPU check verified current RiichiEnv0.4.11 Python/native files against its installed wheel RECORD and replayed all64 cached reference-model games using their recorded actions. Every native event, terminal score and controlled-seat rank was reproduced exactly. This took9.83 seconds, loaded no baseline policy and generated no new baseline-policy games. Preserve the recorded native-file hashes through final comparison; this is additional evidence for cache reuse, not an independent test of alternative trajectories or Mahjong scoring rules. Raw replay details stay private.

An input-isolation check also processed all22,833 training/validation records: public state/instruction/option serialization matched the normal inference path and stayed identical after mutating labels, target distributions, source names and metadata. Fourteen representatives (two per stratum) had identical actual forward-input token arrays across training, inference and mutated-supervision forms, with labels genuinely changed. This excludes those supervision fields as a visible-input shortcut; it does not establish every feature's sufficiency or playing strength. No policy weights were loaded, and registered training code/data stayed unchanged.

After fitting, reload/audit selected and final checkpoints and evaluate the registered selected candidate against the same64 reference games. The completed768 screen is separate because768 can no longer be that selected candidate; no reference-policy rerun is needed. Final-candidate results remain pending; measured384/768 are development checkpoints. Dependency-bound GPU audit job309982 follows a successful fit and is canceled automatically if that dependency fails. Training code8610bff1 passed CI. The reusable evaluator and adverse384 result were published in987ee146; its core and desktop CI jobs44637163535 both passed. Raw records, prediction details and research weights remain private.
