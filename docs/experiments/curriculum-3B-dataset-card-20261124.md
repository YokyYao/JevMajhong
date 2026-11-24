# Prepared 3B data: yaku routes, controls and inherited replay

This card describes the completed data preparation and frozen prospective
recipe, not a trained 3B result. The paired-parent 3B student has not started.
Its parent must be the actual qualified final2400 of the current 3A retry.

| Corpus component | TRAIN decisions | Validation decisions |
|---|---:|---:|
| Exact inherited 3A replay | 87,595 | 23,688 |
| New targeted routes and ordinary controls | 3,652 | 828 |
| Combined prepared corpus | **91,247** | **24,516** |

The new decisions have201 TRAIN and45 validation families. Repeated decisions
along a trajectory are not independent hands. Prepared metadata reports zero
cross-split family/public-input overlap; inherited partitions and replay row
objects remain unchanged. These are development partitions. The reserved final
complete-East seed grid is separate and remains unopened.

| New source pool | TRAIN | Validation | Prospective share of total source loss mass |
|---|---:|---:|---:|
| Kokushi / 国士无双 | 488 | 107 | 3⅓% |
| Chuuren / 九莲宝灯 | 677 | 182 | 3⅓% |
| Suuankou / 四暗刻 | 663 | 147 | 3⅓% |
| Daisangen / 大三元 | 730 | 145 | 3⅓% |
| Chiitoitsu / 七对子 | 732 | 206 | 3⅓% |
| Unselected random-initial-hand control / 普通随机起手对照 | 362 | 41 | 3⅓% |

Together these six pools receive20% of the prospective source loss mass;
the inherited19-source replay receives80%, retaining its relative source
weights. This is weighted loss allocation, not a20% row-sampling rate or a
measured gradient share. Each original TRAIN row appears once per epoch;
no duplicate copies are added. Two epochs give182,494 exposures and2,500
updates at batch1/accumulation73. All25 sources appear in each511-record
mixing block, not necessarily each update. Final accumulation has70 records
per epoch. The same qualified adapter/head continues with a fresh optimizer.

The target route describes how a source hand was constructed; it is neither
an instruction visible to the student nor a forced expert answer. Native
initial hands include different amounts of target-tile replacement (not
measured shanten) and independent physical world variants. Whole canonical-shape families retain
their assigned split. The teacher receives the public observation/history,
without the target tag, hidden opponents' hands, true wall or future events.

Actual Mortal-298k actions supply the new hard labels. A teacher action is
not replaced with the generator's preferred route. The new cohort contains
511 initial-decision labels plus3,969 continuation labels. A separate initial
legal-win shortcut is excluded from this teacher-SFT cohort, rather than
credited as a learned rare-yaku decision. Normalization changes the declared
input representation, preserving actual labels, legal-menu mapping and
family assignments. The prepared encoder checks admitted all4,480 new rows;
the original replay quarantine remains preserved separately.

| New decision context | TRAIN | Validation |
|---|---:|---:|
| Closed-hand discard | 2,504 | 517 |
| Call window | 538 | 128 |
| Open-hand discard | 536 | 151 |
| Kan window | 74 | 32 |

No student capability is established by these counts. In particular,
Mortal agreement is not proof of optimal route choice, and a seven-pairs
source tag does not require pursuing seven pairs. These new3B rows are not
in the currently running3A fit, so current3A behaviour cannot be attributed
to their future20% loss allocation. The prepared recipe does not give seven
pairs a larger source loss mass than the other five new pools.

Actual3B validation will report new-source agreement/loss separately from
ordinary/riichi/call retention and native complete-East outcomes. The small
conditional benchmark uses all92 original validation source cases/45 families,
including the separate win shortcut, against matched frozen-v8 references.
Those biased source-hand outcomes do not estimate natural-game strength or
replace the final independent v8 comparison. Pursuing a route, abandoning it
when appropriate and completing the intended yaku need outcome evidence;
data coverage alone demonstrates none of them.

Prepared combined-data SHA-256 identities:

```text
TRAIN       12a3cef6487c7cdc8a4accab5c4c0bd1e3a5a034dbec82389a2308cfe4f62924
Validation  c77428836780e5b6d68f265f704d39df5f0c7733a13890b75e9e6a7073fa7116
```

Counts and identities above are transcribed from the completed data-v2
preparation receipt and the prepared objective/order summary. This publication
adds no model forwards, optimizer updates, test games or new admission rules.
Raw hands, labels, families, worlds and trajectories remain private.

Generation and label method:
[source scenarios](plans/curriculum-yaku-scenarios-20261123.md),
[actual teacher labels](plans/curriculum-yaku-teacher-20261123.md),
[normalized data preparation](plans/curriculum-yaku-data-v2-20261123.md).
Training and evaluation:
[objective and mixing](plans/curriculum-yaku-course-20261123.md),
[conditional benchmark](plans/curriculum-yaku-validation-cases-20261123.md),
[current conditional launch](plans/curriculum-paired-yaku-continuation-20261124.md).
