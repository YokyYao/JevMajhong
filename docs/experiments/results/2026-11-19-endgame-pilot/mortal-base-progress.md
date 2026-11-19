# Fresh-base Mortal distillation: development progress

Status: training is active. This is a newly initialized rank-16 LoRA and pointer head on pinned Qwen3.5-0.8B, with no Mahjong-trained v8/audited176 adapter or head and no v8 replay targets. The base backbone is frozen. See the [registered training plan](../../plans/mortal-base-training-20261120.md).

GPU job309978 passed preflight and seven checks, encoded all 22,321 teacher training decisions, and completed its first saved-model validation at update64. Both the new LoRA and head changed. Validation uses 512 decisions from 128 whole source games separate from the 384 training games; it is a development set, not an untouched confirmation test.

| Validation metric | Fresh initialization (step0) | Step64 |
|---|---:|---:|
| Weighted teacher cross-entropy (lower is better) | 2.86008 | 1.42608 |
| Weighted teacher agreement | 32.22% | 57.67% |
| Overall agreement on selected 512 menus | 38.67% | 51.37% |
| Exact agreement on 64 teacher-positive chi/pon menus | 98.44% | 0.00% |
| Exact agreement on 64 teacher-negative chi/pon menus | 0.00% | 100.00% |
| Exact agreement on 32 teacher-positive riichi menus | 21.88% | 93.75% |
| Exact agreement on 32 teacher-negative riichi menus | 28.12% | 9.38% |

The overall loss decrease is not evidence that call decisions have been learned. At this early checkpoint the learner has changed from calling too often to always declining calls, while declaring riichi in most available windows. Step0 is a random decision head on a pretrained backbone; it is not an old Mahjong policy or a uniform-random action baseline.

| Available decision in validation | Opportunities | Teacher selects | Step64 selects |
|---|---:|---:|---:|
| Riichi | 64 | 32 | 59 |
| Chi | 75 | 22 | 0 |
| Pon | 76 | 43 | 0 |
| Ankan | 26 | 6 | 0 |
| Daiminkan | 21 | 0 | 0 |
| Kakan | 17 | 3 | 0 |

Opportunity rows overlap: one menu can contain more than one type. Exact positive-action agreement is different from merely choosing that action type. Daiminkan has no positive teacher examples in this data pool; matching its decline decisions does not establish learning when to accept it.

Only 3,072 of 22,321 unique training decisions had been exposed by step64. Follow the fixed schedule: saved validation every64 updates, at least768 updates and full exposure before early stopping, maximum1536. The call-negative population alone needs703 updates at six examples/update for first exposure. No early checkpoint is selected as the final candidate before the observation floor.

After training, reload and audit selected/final checkpoints, evaluate64 new complete East games with the selected candidate, and compare against64 independently audited cached parent games under the identical rules, formatter, guard, seed/seat/opponent grid. A dependency-bound GPU audit job309982 is queued after a successful fit; Slurm cancels it automatically if the dependency fails. There is currently no fresh-base top-two or net-points result, and no claim of improved playing strength. Published code commit `8610bff1cb74d32188c1d356f8961641c098f0cb` passed GitHub CI. Raw records, prediction details and research checkpoints stay private.
