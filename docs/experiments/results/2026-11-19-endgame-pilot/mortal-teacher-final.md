# Mortal teacher continuation: completed development result

The teacher-continuation pilot did not demonstrate improvement in complete East matches. Its best teacher-validation checkpoint was 432 total steps (256 new updates from audited176); training stopped after 384 new updates after four stale checks, beyond the 192-update observation floor and complete exposure to all 1,898 training decisions. The saved selected and final checkpoints reproduced their logged validation through the normal loader.

| Complete East-match metric | Audited176 parent | Teacher continuation, step432 |
|---|---:|---:|
| Games | 64 | 64 |
| Top-two finishes | 31/64 (48.44%) | 27/64 (42.19%) |
| First place | 17.19% | 18.75% |
| Fourth place | 25.00% | 28.12% |
| Mean rank | 2.59375 | 2.671875 |
| Mean net points/game | -1,445.31 | -1,706.25 |
| Hand win rate | 20.63% | 22.32% |
| Deal-in hand rate | 11.17% | 13.33% |

The paired top-two difference is -6.25 percentage points. The whole-seed 95% t interval is [-28.59, +16.09] percentage points across eight seed groups. This small development grid cannot establish either improvement or definitive degradation. Net points decreased by 260.94 per game. More wins coexisted with more deal-ins; action counts alone do not establish why.

Teacher-validation macro cross-entropy improved from .73647 to .49796 and overall agreement from 66.80% to 73.05%. These auxiliary gains did not translate into a positive match result. The data selected only 1,024 teacher training menus plus 874 retained decisions. Its positive-action classes also distorted select/decline sampling: the 20 riichi validation opportunities contained 16 teacher positives, and early students chose riichi in all 20. This motivates an opportunity-based sampling redesign, not a claim that Mortal imitation necessarily lowers playing strength.

Independent checks reconciled all 128 complete evaluation game logs and 512 source games for payments, riichi-stick pool conservation and terminal ranks. This was an independent accounting/statistics audit, not a separate han/fu rules engine. Matches used eight seeds across all four controlled seats and two opponent profiles; initial seeds were paired, while later trajectories could diverge. The grid is inspected development evidence, not independent confirmation. Source games were split 384/128 by whole game.

The previously released model and EXE were unchanged. The next authorized experiment is [teacher-only training starting from the Qwen base](../../plans/mortal-base-training-20261120.md), with a new LoRA/head, all 22,321 admitted teacher TRAIN menus and explicit select/decline strata. Existing raw records, predictions and research checkpoints remain private; this report publishes aggregates only.
