# Kan label coverage and cached decisions

All counts come from the frozen training/validation corpora and already saved validation probabilities. No inference or new games were run. Source preference differences are not proof of contradictory labels on identical states.

## Training coverage by source

| Source | Operation | Legal menus | Hard positive | Hard negative | Soft preferred | Soft other preferred | Soft tied | Soft unlabelled | Soft no other |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| curriculum2:advanced:legacy:kan | ankan | 895 | 445 | 450 | 0 | 0 | 0 | 0 | 0 |
| curriculum2:advanced:legacy:kan | daiminkan | 662 | 341 | 321 | 0 | 0 | 0 | 0 | 0 |
| curriculum2:advanced:legacy:kan | kakan | 231 | 110 | 121 | 0 | 0 | 0 | 0 | 0 |
| curriculum2:advanced:teacher:kan | ankan | 299 | 56 | 243 | 0 | 0 | 0 | 0 | 0 |
| curriculum2:advanced:teacher:kan | daiminkan | 281 | 0 | 281 | 0 | 0 | 0 | 0 | 0 |
| curriculum2:advanced:teacher:kan | kakan | 149 | 47 | 102 | 0 | 0 | 0 | 0 | 0 |
| curriculum3:mc:kan_window | ankan | 98 | 0 | 0 | 18 | 66 | 14 | 0 | 0 |
| curriculum3:mc:kan_window | daiminkan | 109 | 0 | 0 | 18 | 77 | 14 | 0 | 0 |
| curriculum3:mc:kan_window | kakan | 41 | 0 | 0 | 12 | 22 | 7 | 0 | 0 |
| curriculum3B:teacher:chiitoitsu | ankan | 2 | 1 | 1 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:chiitoitsu | daiminkan | 2 | 0 | 2 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:chiitoitsu | kakan | 3 | 1 | 2 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:chuuren | ankan | 25 | 1 | 24 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:chuuren | daiminkan | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:chuuren | kakan | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:daisangen | ankan | 5 | 5 | 0 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:daisangen | daiminkan | 8 | 0 | 8 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:daisangen | kakan | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:kokushi | daiminkan | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:kokushi | kakan | 3 | 0 | 3 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:suuankou | ankan | 6 | 5 | 1 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:suuankou | daiminkan | 4 | 0 | 4 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:unselected_random_initial_hand_control | ankan | 10 | 1 | 9 | 0 | 0 | 0 | 0 | 0 |
| curriculum3B:teacher:unselected_random_initial_hand_control | daiminkan | 1 | 0 | 1 | 0 | 0 | 0 | 0 | 0 |

## Validation selection by label category

| Operation | Label category | Menus | Selected 0 | Selected 625 | Selected 1250 |
| --- | --- | ---: | ---: | ---: | ---: |
| ankan | hard_positive | 152 | 83/152 (54.61%) | 78/152 (51.32%) | 88/152 (57.89%) |
| ankan | hard_negative | 227 | 70/227 (30.84%) | 75/227 (33.04%) | 77/227 (33.92%) |
| ankan | soft_preferred | 25 | 4/25 (16.00%) | 5/25 (20.00%) | 4/25 (16.00%) |
| ankan | soft_other_preferred | 52 | 17/52 (32.69%) | 19/52 (36.54%) | 18/52 (34.62%) |
| ankan | soft_tied | 9 | 3/9 (33.33%) | 3/9 (33.33%) | 4/9 (44.44%) |
| daiminkan | hard_positive | 88 | 4/88 (4.55%) | 9/88 (10.23%) | 4/88 (4.55%) |
| daiminkan | hard_negative | 197 | 6/197 (3.05%) | 18/197 (9.14%) | 5/197 (2.54%) |
| daiminkan | soft_preferred | 12 | 1/12 (8.33%) | 2/12 (16.67%) | 2/12 (16.67%) |
| daiminkan | soft_other_preferred | 35 | 0/35 (0.00%) | 4/35 (11.43%) | 1/35 (2.86%) |
| daiminkan | soft_tied | 4 | 0/4 (0.00%) | 0/4 (0.00%) | 0/4 (0.00%) |
| kakan | hard_positive | 72 | 30/72 (41.67%) | 36/72 (50.00%) | 30/72 (41.67%) |
| kakan | hard_negative | 78 | 34/78 (43.59%) | 40/78 (51.28%) | 36/78 (46.15%) |
| kakan | soft_preferred | 6 | 2/6 (33.33%) | 2/6 (33.33%) | 2/6 (33.33%) |
| kakan | soft_other_preferred | 7 | 1/7 (14.29%) | 2/7 (28.57%) | 1/7 (14.29%) |

Hard positives/negatives follow cached action-type labels. Soft preferred/other/tied compares the largest labelled candidate probability of that operation with the largest labelled alternative. Legal but unlabelled operations are separate; they are not negative labels. None of these categories proves real kan profitability. The CSV also reports source masses: these describe sampling/loss allocation, not realized reward.

Active training, its 2500-update budget, dataset and final confirmation remain unchanged. Raw hands and probabilities stay private.
