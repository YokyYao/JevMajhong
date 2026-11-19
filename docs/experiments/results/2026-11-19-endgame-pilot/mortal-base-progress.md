# Fresh-base Mortal distillation: development progress

Status: training is active. This is a new rank-16 LoRA and pointer head on pinned Qwen3.5-0.8B, with no Mahjong-trained v8/audited176 adapter or head and no v8 replay targets. The backbone is frozen. See the [registered plan](../../plans/mortal-base-training-20261120.md).

GPU job309978 passed preflight and seven checks and encoded all22,321 teacher training decisions. Saved-model validation is complete through update192; both LoRA and head changed. The512 validation decisions come from128 whole source games separate from the384 training games. This is an inspected development set, not an untouched confirmation test.

| Saved step | Weighted teacher CE ↓ | Weighted agreement | Overall agreement, selected512 menus | Unique training decisions exposed |
|---|---:|---:|---:|---:|
| 0 | 2.86008 | 32.22% | 38.67% | 0 |
| 64 | 1.42608 | 57.67% | 51.37% | 3,072 |
| 128 | 1.13246 | 61.30% | 54.49% | 6,144 |
| 192 | 1.02006 | 64.55% | 59.77% | 9,033 |

Step0 is a random decision head on a pretrained backbone, not an old Mahjong policy or a uniform-random action baseline. Agreement depends on this selected/weighted validation mix and is not a win rate.

| Saved step | Exact teacher-positive chi/pon matches /64 | Exact teacher-negative chi/pon matches /64 | Exact teacher-positive riichi matches /32 | Exact teacher-negative riichi matches /32 |
|---|---:|---:|---:|---:|
| 0 | 63 | 0 | 7 | 9 |
| 64 | 0 | 64 | 30 | 3 |
| 128 | 0 | 64 | 26 | 11 |
| 192 | 16 | 64 | 23 | 15 |

The initial learner changed from calling too often to always declining chi/pon at64–128, then began selecting some teacher-positive calls at192. Most teacher-positive calls are still missed. No claim of solved call decisions or improved playing strength follows from the overall CE decrease.

| Available decision | Validation opportunities | Teacher selects | Step128 selects | Step192 selects |
|---|---:|---:|---:|---:|
| Riichi | 64 | 32 | 44 | 39 |
| Chi | 75 | 22 | 0 | 6 |
| Pon | 76 | 43 | 0 | 11 |
| Ankan | 26 | 6 | 16 | 3 |
| Daiminkan | 21 | 0 | 0 | 0 |
| Kakan | 17 | 3 | 4 | 1 |

Opportunity rows overlap: a menu can contain several action types. At192, all6 selected chi and11 selected pon match the teacher's action type; exact candidate agreement is16/64 on the teacher-positive call stratum. These are teacher matches, not independent expert correctness labels.

Riichi at192 has23 teacher-positive matches,16 extra declarations,9 misses and16 correct declines. Ankan has2 matches,1 extra and4 misses; kakan has0 matches,1 extra and3 misses. Reduced kan frequency therefore does not establish better kan judgment. At128, ankan had5 matches and11 extras; behavior changed substantially while overall kan-menu exact agreement remained62.5%.

## Teacher daiminkan bridge diagnostic

The source pool has no positive daiminkan labels. To check the interface, a bounded diagnostic inside the existing GPU allocation replayed4 complete stored training-source games. All4 inspected controlled-player E1–3 daiminkan windows exposed the kan action in the teacher's real legal mask, but the teacher's natural Q ranking preferred another action. Two separate Bot copies were forced to select kan solely to test decoding: each produced daiminkan and mapped uniquely into the native legal menu. The real source trajectories remained identical to the stored logs; no forced action was applied to a source game or added to training.

This limited diagnostic checks visibility and decoding in those windows, not every source window and not whether the teacher's preferences maximize Mahjong results. The absence of positive labels remains a training-data limitation; matching declines cannot establish learned acceptance of daiminkan. Detailed source events/Q values remain private.

## Schedule and integrity

At192, only9,033/22,321 unique training decisions have been exposed. Continue the fixed schedule: saved validation every64 updates, at least768 updates and full exposure before early stopping, maximum1536. The call-negative population alone needs703 updates at six examples/update for first exposure. No early checkpoint is selected as final before the observation floor.

A read-only companion-CPU step inside the same GPU allocation verified all10 base snapshot files (1,769,905,646 bytes) against their original content addresses, including weights and tokenizer. Nested links to shared storage are handled using the snapshot's content address. This is local pinned-cache integrity, not a new remote signature. Registered training sources/data match their frozen hashes, and eight published source/plan/report files match their Git blob identities. Post-fit checks were compiled and frozen; they repeat base integrity checking after fitting.

After fitting, reload/audit selected and final checkpoints, run64 new complete East games for the selected candidate, and compare with64 independently audited cached parent games using identical rules, formatter, guard and seed/seat/opponent grid. Dependency-bound GPU audit job309982 follows a successful fit and is canceled automatically if that dependency fails. No fresh-base top-two or net-points result exists yet. Code commit 8610bff1cb74d32188c1d356f8961641c098f0cb passed GitHub CI; subsequent progress updates change only this aggregate report. Raw records, prediction details and research checkpoints remain private.
