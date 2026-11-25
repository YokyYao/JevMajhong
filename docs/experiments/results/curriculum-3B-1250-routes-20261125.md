# Observed 3B route and retained-operation diagnostics

These observations reuse completed full validation. No additional model inference, games or updates were run.

## Route teacher labels

The 828 route/control validation menus come from 45 source families; repeated decisions are not independent games. CE measures probability assigned to the teacher action, and agreement measures the selected action matching that label. Neither proves that the model can complete the named yaku in real play.

| Route | Train menus | Validation menus | CE 0 | CE 625 | CE 1250 | Agreement 0 | Agreement 1250 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Kokushi musou | 488 | 107 | 1.25869 | 1.24723 | 1.15167 | 51.40% | 56.07% |
| Chuuren poutou | 677 | 182 | 0.95863 | 0.89800 | 0.86520 | 65.93% | 69.23% |
| Suuankou | 663 | 147 | 0.87943 | 0.87575 | 0.84975 | 64.63% | 65.99% |
| Daisangen | 730 | 145 | 0.84434 | 0.75637 | 0.72568 | 66.21% | 75.86% |
| Chiitoitsu | 732 | 206 | 0.82986 | 0.83690 | 0.79893 | 71.36% | 69.42% |
| Random initial-hand control | 362 | 41 | 0.56316 | 0.57955 | 0.55922 | 80.49% | 80.49% |

## Retained operation labels

The positive and negative menus are the unchanged original-call validation subset, excluding the new route menus. Accept recall means selecting the operation when the cached teacher/legacy label selects it; decline recall means declining when that label does. These are imitation diagnostics, not expert judgments about profitable calls.

| Operation | Positive menus | Negative menus | Accept 0 | Accept 1250 | Decline 0 | Decline 1250 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reach | 334 | 326 | 74.55% | 61.68% | 58.28% | 73.62% |
| chi | 415 | 2622 | 47.71% | 54.70% | 97.03% | 94.66% |
| pon | 561 | 1430 | 67.20% | 78.79% | 91.12% | 87.27% |
| ankan | 151 | 206 | 54.30% | 57.62% | 68.45% | 65.05% |
| daiminkan | 88 | 191 | 4.55% | 4.55% | 96.86% | 97.38% |
| kakan | 71 | 75 | 42.25% | 42.25% | 54.67% | 56.00% |

The course and original 2500-update budget remain unchanged. This report does not impose an intermediate grade veto, select a checkpoint, open reserved confirmation seeds or replace production weights. Raw examples stay private.
