# Fresh-base Mortal distillation: development progress

Status: training is active. This is a newly initialized rank-16 LoRA and pointer head on pinned Qwen3.5-0.8B, with no Mahjong-trained v8/audited176 adapter or head and no v8 replay targets. The base backbone is frozen. See the [registered training plan](../../plans/mortal-base-training-20261120.md).

GPU job309978 passed preflight and seven checks, encoded all 22,321 teacher training decisions, and completed saved-model validations at updates64 and128. Both the new LoRA and head changed. Validation uses 512 decisions from 128 whole source games separate from the 384 training games; it is a development set, not an untouched confirmation test.

| Validation metric | Fresh initialization (step0) | Step64 | Step128 |
|---|---:|---:|---:|
| Weighted teacher cross-entropy (lower is better) | 2.86008 | 1.42608 | 1.13246 |
| Weighted teacher agreement | 32.22% | 57.67% | 61.30% |
| Overall agreement on selected 512 menus | 38.67% | 51.37% | 54.49% |
| Exact agreement on 64 teacher-positive chi/pon menus | 98.44% | 0.00% | 0.00% |
| Exact agreement on 64 teacher-negative chi/pon menus | 0.00% | 100.00% | 100.00% |
| Exact agreement on 32 teacher-positive riichi menus | 21.88% | 93.75% | 81.25% |
| Exact agreement on 32 teacher-negative riichi menus | 28.12% | 9.38% | 34.38% |

The overall loss decrease is not evidence that chi/pon decisions have been learned. The early learner changed from calling too often to always declining chi/pon. At128, excessive riichi is reduced and kan decisions have changed, but chi/pon positive recall is still zero. Step0 is a random decision head on a pretrained backbone; it is not an old Mahjong policy or a uniform-random action baseline.

| Available decision in validation | Opportunities | Teacher selects | Step64 selects | Step128 selects |
|---|---:|---:|---:|---:|
| Riichi | 64 | 32 | 59 | 44 |
| Chi | 75 | 22 | 0 | 0 |
| Pon | 76 | 43 | 0 | 0 |
| Ankan | 26 | 6 | 0 | 16 |
| Daiminkan | 21 | 0 | 0 | 0 |
| Kakan | 17 | 3 | 0 | 4 |

Opportunity rows overlap: one menu can contain more than one type. Exact positive-action agreement is different from merely choosing that action type. Daiminkan has no positive teacher examples in this data pool; matching its decline decisions does not establish learning when to accept it.

At128, riichi has26 teacher-positive matches,18 teacher-negative extra declarations,6 missed declarations and14 correct declines. Ankan has5 positive matches but11 extra declarations relative to the teacher; kakan has2 positive matches,2 extras and1 miss. These are comparisons with teacher choices, not independently proven correct or incorrect Mahjong decisions. In particular, executing kan more often does not by itself prove better judgment.

Only 3,072 of 22,321 unique training decisions had been exposed by step64, and6,144 by step128. Follow the fixed schedule: saved validation every64 updates, at least768 updates and full exposure before early stopping, maximum1536. The call-negative population alone needs703 updates at six examples/update for first exposure. No early checkpoint is selected as the final candidate before the observation floor.

A read-only companion-CPU step inside the same GPU allocation verified all10 base snapshot files (1,769,905,646 bytes) against their content addresses, including the model weights and tokenizer. The cache uses nested links to shared storage; verification follows the snapshot's original content address, not the shared-storage filename. This checks local pinned-cache integrity, not a new remote signature. The post-fit audit repeats this check.

After training, reload and audit selected/final checkpoints, evaluate64 new complete East games with the selected candidate, and compare against64 independently audited cached parent games under the identical rules, formatter, guard, seed/seat/opponent grid. A dependency-bound GPU audit job309982 is queued after a successful fit; Slurm cancels it automatically if the dependency fails. There is currently no fresh-base top-two or net-points result, and no claim of improved playing strength. Published code commit `8610bff1cb74d32188c1d356f8961641c098f0cb` passed GitHub CI. Raw records, prediction details and research checkpoints stay private.
