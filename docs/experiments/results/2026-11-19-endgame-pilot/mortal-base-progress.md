# Fresh-base Mortal distillation: development progress

Status: training is active. This is a new rank-16 LoRA and pointer head on pinned Qwen3.5-0.8B, with no Mahjong-trained v8/audited176 adapter or head and no v8 replay targets. The backbone is frozen. See the [registered plan](../../plans/mortal-base-training-20261120.md).

GPU job309978 passed preflight and seven checks and encoded all22,321 teacher training decisions. Saved-model validation is complete through update384; both LoRA and head changed. The512 validation decisions come from128 whole source games separate from the384 training games. This is an inspected development set, not an untouched confirmation test. This pure-teacher phase covers East1–3 non-win menus under the existing serving guard; East4 pursuit and win acceptance receive no direct teacher supervision here, so agreement cannot establish those abilities.

| Saved step | Weighted teacher CE ↓ | Weighted agreement | Overall agreement, selected512 menus | Unique training decisions exposed |
|---|---:|---:|---:|---:|
| 0 | 2.86008 | 32.22% | 38.67% | 0 |
| 64 | 1.42608 | 57.67% | 51.37% | 3,072 |
| 128 | 1.13246 | 61.30% | 54.49% | 6,144 |
| 192 | 1.02006 | 64.55% | 59.77% | 9,033 |
| 256 | 0.89589 | 67.59% | 63.67% | 11,467 |
| 320 | 0.86502 | 67.03% | 64.84% | 13,899 |
| 384 | 0.83625 | 70.79% | 68.36% | 16,095 |

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

The initial learner changed from calling too often to always declining chi/pon at64–128, then began selecting some teacher-positive calls at192. Exact positive-call matches increase16/64 →22/64 →34/64 →37/64 at192/256/320/384, while negative-call matches change64/64 →64/64 →62/64 →62/64. Higher call recall now comes with two extra calls relative to the teacher. No claim of solved call decisions or improved playing strength follows from the overall CE decrease. Weighted agreement at320 was slightly below256 despite lower CE.

| Available decision | Validation opportunities | Teacher selects | Step320 selects | Step384 selects |
|---|---:|---:|---:|---:|
| Riichi | 64 | 32 | 33 | 48 |
| Chi | 75 | 22 | 11 | 11 |
| Pon | 76 | 43 | 26 | 29 |
| Ankan | 26 | 6 | 9 | 0 |
| Daiminkan | 21 | 0 | 0 | 0 |
| Kakan | 17 | 3 | 2 | 1 |

Opportunity rows overlap: a menu can contain several action types. At384,11/11 selected chi and27/29 selected pon match the teacher's action type; exact candidate agreement is37/64 on the teacher-positive call stratum. These are teacher matches, not independent expert correctness labels.

Riichi at384 has29 teacher-positive matches,19 extra declarations,3 misses and13 correct declines. Its count rises33 →48 after nearly matching the teacher's32 at320; the earlier similar count did not mean it chose the same situations. Ankan now has0 matches,0 extras and6 misses; kakan has0 matches,1 extra and3 misses. Kan judgment remains insufficient: overall kan-menu exact agreement is64.06% at320/384 versus68.75% at256, while ankan selections change2 →9 →0. Lower overall CE is not evidence of solved kan decisions.

## Teacher daiminkan bridge diagnostic

The source pool has no positive daiminkan labels. To check the interface, a bounded diagnostic inside the existing GPU allocation replayed4 complete stored training-source games. All4 inspected controlled-player E1–3 daiminkan windows exposed the kan action in the teacher's real legal mask, but the teacher's natural Q ranking preferred another action. Two separate Bot copies were forced to select kan solely to test decoding: each produced daiminkan and mapped uniquely into the native legal menu. The real source trajectories remained identical to the stored logs; no forced action was applied to a source game or added to training.

This limited diagnostic checks visibility and decoding in those windows, not every source window and not whether the teacher's preferences maximize Mahjong results. The absence of positive labels remains a training-data limitation; matching declines cannot establish learned acceptance of daiminkan. Detailed source events/Q values remain private.

## Schedule and integrity

At384, only16,095/22,321 unique training decisions have been exposed. Continue the fixed schedule: saved validation every64 updates, at least768 updates and full exposure before early stopping, maximum1536. The call-negative population alone needs703 updates at six examples/update for first exposure. No early checkpoint is selected as final before the observation floor.

A read-only companion-CPU step inside the same GPU allocation verified all10 base snapshot files (1,769,905,646 bytes) against their original content addresses, including weights and tokenizer. Nested links to shared storage are handled using the snapshot's content address. This is local pinned-cache integrity, not a new remote signature. Registered training sources/data match their frozen hashes, and eight published source/plan/report files match their Git blob identities. Post-fit checks were compiled and frozen; they repeat base integrity checking after fitting.

A separate bounded companion-CPU check verified current RiichiEnv0.4.11 Python/native files against its installed wheel RECORD and replayed all64 cached reference-model games using their recorded actions. Every native event, terminal score and controlled-seat rank was reproduced exactly. This took9.83 seconds, loaded no baseline policy and generated no new baseline-policy games. Preserve the recorded native-file hashes through final comparison; this is additional evidence for cache reuse, not an independent test of alternative trajectories or Mahjong scoring rules. Raw replay details stay private.

An input-isolation check also processed all22,833 training/validation records: public state/instruction/option serialization matched the normal inference path and stayed identical after mutating labels, target distributions, source names and metadata. Fourteen representatives (two per stratum) had identical actual forward-input token arrays across training, inference and mutated-supervision forms, with labels genuinely changed. This excludes those supervision fields as a visible-input shortcut; it does not establish every feature's sufficiency or playing strength. No policy weights were loaded, and registered training code/data stayed unchanged.

After fitting, reload/audit selected and final checkpoints, run64 new complete East games for the selected candidate, and compare with64 independently audited cached parent games using identical rules, formatter, guard and seed/seat/opponent grid. Dependency-bound GPU audit job309982 follows a successful fit and is canceled automatically if that dependency fails. No fresh-base top-two or net-points result exists yet. Code commit 8610bff1cb74d32188c1d356f8961641c098f0cb passed GitHub CI; subsequent progress updates change only this aggregate report. Raw records, prediction details and research checkpoints remain private.
