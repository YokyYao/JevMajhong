# Mortal teacher: isolated data complete, fit launched

The user requested training with the already available local
[VoidShine/mortal-298k](https://huggingface.co/VoidShine/mortal-298k).
The [training plan](../../plans/mortal-teacher-20261119.md) and
[pre-fit validation addendum](../../plans/mortal-teacher-validation-addendum-20261119.md)
are registered before optimizer updates. Formal model/main/Windows remain unchanged.

GPU309748 on ws-l2-016 completed512 native Mahjong Soul East source games:
384 TRAIN sources and128 disjoint VALIDATION sources. Every match terminated
and its score ledger reconciled. Teacher received only its own player-visible
events; concealed opponent hands/draws and ambiguous/illegal mappings abort.
Strict local teacher weight and native-extension checks passed, with no rule
fallback. This uses the third-party298k teacher, not unpublished official weights.

Outcome-blind selection produced the following exact quotas. At most16 selected
menus come from any one game. Selected TRAIN/VALIDATION menus span341/108 distinct
games respectively; the remaining generated games did not contribute selected
menus. Whole-game source identities are disjoint.

| Source | TRAIN | VALIDATION | Loss mass |
|---|---:|---:|---:|
|Teacher ordinary discard|512|128|25%|
|Teacher discard under riichi threat|128|32|10%|
|Teacher riichi|64|16|5%|
|Teacher chi|64|16|7.5%|
|Teacher pon|64|16|7.5%|
|Teacher decline-call controls|128|32|10%|
|Teacher kan windows|64|16|5%|
|Retained ordinary/leading replay|608|old validation retained|15%|
|Retained endgame targets|266|old validation retained|15%|

Totals:1898 positive TRAIN rows (1024 new plus874 retained),256 new held-out
teacher menus. The old186 validation rows and8 fresh endgame monitoring menus
remain separate and unchanged. Teacher labels use mapped choices with2% smoothing,
not calibrated win probabilities. Teacher imitation excludes East4/extensions,
win/pass menus and choices outside the unchanged Jev serving guard.

Selected TRAIN kan windows include28 ankan,26 daiminkan and10 kakan opportunities;
VALIDATION includes5/10/1 respectively. Teacher selected9 ankan and2 kakan in
TRAIN,1 each in VALIDATION, and ZERO daiminkan. Other choices include discard,
pass, riichi, pon and abortive draw. Thus teacher supervision alone is not evidence
that the model will learn profitable daiminkan. The single kakan validation menu
is also insufficient for a broad kakan-strength claim. Per-type choose/pass
metrics are required alongside overall teacher accuracy.

Data job309748 exited0. Data source5d95c97; validation addendum source147da6a;
corrected label-adapter trainer source4851d2a.
All290 prior core tests passed GPU309751; its desktop invocation initially failed
only because a headless Qt backend was not set. Desktop rerun309752 passed89
checks with1 optional skip. Six updated trainer checks passed GPU309760;
GitHub CI44601100607 passed. A one-shot companion-CPU step inside GPU309748 froze
109 source/config/plan files before fitting. No model computation ran on login.

Fit309763 started on the same clean GPU after successful data completion, but
exited1 on its FIRST encoding operation before any parameter update: the pinned
Jev materializer also requires an explicit label field. The verified teacher
action already exists in metadata. The [label adapter correction](../../plans/mortal-teacher-label-adapter-20261119.md)
adds that field to an encoding-only COPY, preserving every frozen data byte and
target. All1898 actual TRAIN and256 VALIDATION records must pass the real pinned
materializer in a GPU job before a replacement fit starts. Failed metadata and
the exit1 receipt are preserved; no trained weight is attributed to309763.
Recovery GPU309774 passed7 updated trainer checks and all2154 ACTUAL records
(1898 TRAIN+256 VALIDATION) through pinned Jev materialize, preserving data hashes.
Source4851d2a passed CI44602844235. A113-file pre-fit freeze includes pinned
upstream data/model/checkpoint modules. Recovery fit309775 is RUNNING on clean
ws-l2-016 and has completed its first4 real optimizer updates (total180).
Old pending309756 was
canceled before computation for the pre-fit addendum. Finished checks aged out
of Slurm; their successful exit receipts and the frozen source proof are
verified by the launch script rather than invalid scheduler dependencies.

Start from exact176 LoRA/head; frozen backbone, new AdamW, existing model only.
Observe at least192 new48-example accumulations with complete data exposure;
maximum512, validate normally loaded saved weights every32, then four-stale-check
stopping. The initial saved-and-normal-loaded176 control reproduced endgame
primary .23514093137254902. On256 held-out teacher menus its macro CE is
.7364710662852886, macro action agreement64.17%, overall agreement66.80%.
These are teacher-agreement metrics, NOT match win/top2 rates. In the10 held-out
daiminkan windows, teacher chose9 passes and1 pon; parent chose5 passes and5
daiminkan. Improving teacher agreement can therefore reduce daiminkan choices,
and cannot itself establish better kan strategy. No nonzero checkpoint outcome
gain is claimed before the first32-update validation.
The planned comparison is candidate versus exact176 parent, each64 COMPLETE
East games, including a Mortal-opponent profile, with whole-seed95% intervals.
If no checkpoint clears the screen, compare the best nonzero teacher-CE checkpoint
as an explicitly screen-ineligible diagnostic rather than skipping match evidence.

Only code/plans/aggregate findings are public. Raw games, traces, labels/features
and research weights remain private. No independent confirmation or release claim.
