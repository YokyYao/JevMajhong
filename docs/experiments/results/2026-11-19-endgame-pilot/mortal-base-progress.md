# Fresh-base Mortal distillation: development progress

Status: training is active. This is a new rank-16 LoRA and pointer head on pinned Qwen3.5-0.8B, with no Mahjong-trained v8/audited176 adapter or head and no v8 replay targets. The backbone is frozen. See the [registered plan](../../plans/mortal-base-training-20261120.md).

GPU job309978 passed preflight and seven checks and encoded all22,321 teacher training decisions. Saved-model validation is complete through update256; both LoRA and head changed. The512 validation decisions come from128 whole source games separate from the384 training games. This is an inspected development set, not an untouched confirmation test.

| Saved step | Weighted teacher CE ↓ | Weighted agreement | Overall agreement, selected512 menus | Unique training decisions exposed |
|---|---:|---:|---:|---:|
| 0 | 2.86008 | 32.22% | 38.67% | 0 |
| 64 | 1.42608 | 57.67% | 51.37% | 3,072 |
| 128 | 1.13246 | 61.30% | 54.49% | 6,144 |
| 192 | 1.02006 | 64.55% | 59.77% | 9,033 |
| 256 | 0.89589 | 67.59% | 63.67% | 11,467 |

Step0 is a random decision head on a pretrained backbone, not an old Mahjong policy or a uniform-random action baseline. Agreement depends on this selected/weighted validation mix and is not a win rate.

| Saved step | Exact teacher-positive chi/pon matches /64 | Exact teacher-negative chi/pon matches /64 | Exact teacher-positive riichi matches /32 | Exact teacher-negative riichi matches /32 |
|---|---:|---:|---:|---:|
| 0 | 63 | 0 | 7 | 9 |
| 64 | 0 | 64 | 30 | 3 |
| 128 | 0 | 64 | 26 | 11 |
| 192 | 16 | 64 | 23 | 15 |
| 256 | 22 | 64 | 29 | 11 |

The initial learner changed from calling too often to always declining chi/pon at64–128, then began selecting some teacher-positive calls at192. Exact positive-call matches increase from16/64 at192 to22/64 at256 while exact negative-call matches stay64/64. Most teacher-positive calls are still missed. No claim of solved call decisions or improved playing strength follows from the overall CE decrease.

| Available decision | Validation opportunities | Teacher selects | Step192 selects | Step256 selects |
|---|---:|---:|---:|---:|
| Riichi | 64 | 32 | 39 | 49 |
| Chi | 75 | 22 | 6 | 9 |
| Pon | 76 | 43 | 11 | 14 |
| Ankan | 26 | 6 | 3 | 2 |
| Daiminkan | 21 | 0 | 0 | 0 |
| Kakan | 17 | 3 | 1 | 1 |

Opportunity rows overlap: a menu can contain several action types. At256, all9 selected chi and14 selected pon match the teacher's action type; exact candidate agreement is22/64 on the teacher-positive call stratum. These are teacher matches, not independent expert correctness labels.

Riichi at256 has29 teacher-positive matches,20 extra declarations,3 misses and12 correct declines. Its selection count rises again from39/64 to49/64 despite falling total validation loss. Ankan has1 match,1 extra and5 misses; kakan has0 matches,1 extra and3 misses. Reduced kan frequency therefore does not establish better kan judgment. At128, ankan had5 matches and11 extras; behavior changed substantially while overall kan-menu exact agreement at128/192 remained62.5% (68.75% at256).

## Teacher daiminkan bridge diagnostic

The source pool has no positive daiminkan labels. To check the interface, a bounded diagnostic inside the existing GPU allocation replayed4 complete stored training-source games. All4 inspected controlled-player E1–3 daiminkan windows exposed the kan action in the teacher's real legal mask, but the teacher's natural Q ranking preferred another action. Two separate Bot copies were forced to select kan solely to test decoding: each produced daiminkan and mapped uniquely into the native legal menu. The real source trajectories remained identical to the stored logs; no forced action was applied to a source game or added to training.

This limited diagnostic checks visibility and decoding in those windows, not every source window and not whether the teacher's preferences maximize Mahjong results. The absence of positive labels remains a training-data limitation; matching declines cannot establish learned acceptance of daiminkan. Detailed source events/Q values remain private.

## Schedule and integrity

At256, only11,467/22,321 unique training decisions have been exposed. Continue the fixed schedule: saved validation every64 updates, at least768 updates and full exposure before early stopping, maximum1536. The call-negative population alone needs703 updates at six examples/update for first exposure. No early checkpoint is selected as final before the observation floor.

A read-only companion-CPU step inside the same GPU allocation verified all10 base snapshot files (1,769,905,646 bytes) against their original content addresses, including weights and tokenizer. Nested links to shared storage are handled using the snapshot's content address. This is local pinned-cache integrity, not a new remote signature. Registered training sources/data match their frozen hashes, and eight published source/plan/report files match their Git blob identities. Post-fit checks were compiled and frozen; they repeat base integrity checking after fitting.

A separate bounded companion-CPU check verified current RiichiEnv0.4.11 Python/native files against its installed wheel RECORD and replayed all64 cached reference-model games using their recorded actions. Every native event, terminal score and controlled-seat rank was reproduced exactly. This took9.83 seconds, loaded no baseline policy and generated no new baseline-policy games. Preserve the recorded native-file hashes through final comparison; this is additional evidence for cache reuse, not an independent test of alternative trajectories or Mahjong scoring rules. Raw replay details stay private.

After fitting, reload/audit selected and final checkpoints, run64 new complete East games for the selected candidate, and compare with64 independently audited cached parent games using identical rules, formatter, guard and seed/seat/opponent grid. Dependency-bound GPU audit job309982 follows a successful fit and is canceled automatically if that dependency fails. No fresh-base top-two or net-points result exists yet. Code commit 8610bff1cb74d32188c1d356f8961641c098f0cb passed GitHub CI; subsequent progress updates change only this aggregate report. Raw records, prediction details and research checkpoints remain private.
