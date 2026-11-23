# Stage 3B launch after the actual final 3A

Status: launch code prepared; parent not selected and student updates not started.

This continues the fresh-base curriculum, rather than v8. The final 3A step2400 is
the only prospective parent (cumulative3396 updates). Its full independent audit
and its original development advancement grade must pass. If the grade fails,
the preparation fails visibly and dependent jobs are cancelled; the grade is
never overwritten. A research continuation after a failed grade requires a
separate documented decision and configuration, not automatic admission.

## Fixed training

The already normalized corpus has91247 TRAIN rows and24516 validation rows.
Every original87595/23688 replay row remains in its original split. There are
3652/828 new hard-labelled decisions from201/45 targeted scenario families.
Mortal labels are decisions observed from native play, not expert optimality.

The fixed loss mass is80% replay (same nineteen-source relative shares) and20%
equally divided across kokushi, chuuren, suuankou, daisangen, chiitoitsu and random
initial-hand control. This is scalar loss weighting, not physical row duplication.
Two complete without-replacement epochs mean182494 row exposures and2500 updates
with batch1/accumulation73. Each511-record mixing block includes all25 sources.
The final update in each epoch has70 records. Learning rates are3e-6 and1.5e-5.
The same LoRA and pointer head continue with a new optimizer; no residual network
is added, no base is silently changed and no v8 weights are loaded as a parent.

## Curves and outcomes

Actual student validation and complete East matches are saved at
0/625/1250/1875/2500. Pre-update checks require372 adapter tensors and4 head tensors
equal to the actual parent, all23688 parent validation menus reproduced within
5e-5 with identical decisions, and the same sixteen full parent games reproduced.
Metadata serialization may differ only with these numerical proofs.

Ordinary optimization does not stop on an intermediate noisy game point:2500 is
the registered final candidate, with no minimum-validation-loss peak selection.
Fatal numerical/runtime/inheritance failures remain errors. Real train loss and
CE are recorded every update; reward is null for SFT updates. Game reward,
hand-win/top2/net/dealin come only from actual completed matches. The first four
curve points use16 games each, the final uses64 and a separately reported same16
subset. Existing parent, old guarded stage2 and frozen v8 caches are reused.

At step0 and2500 the same92 validation scenario cases/45 families are played to
the first real settlement against three fixed stateful Mortal opponents.
The frozen v8 conditional cache is reused. These biased first-hand outcomes and
supported yaku counts are reported separately from natural complete-match rates;
an initial legal-win shortcut and unsupported scoring cases stay explicit.
Source route and hidden wall are private provenance, never model inputs.

The final audit checks all2500 optimizer observations, all376 Adam states,
two exact epoch orders, saved/reloaded24516 menus, actual match ledgers,
and conditional outcomes. Development targets compare retention and native
outcomes with the actual parent and six-source CE improvement with step0.
The audit can pass as evidence while the separate strategy advancement grade
fails; these must never be conflated.

## Resource and publication rules

All compilation, tests, hashing, data reads and model work run inside bounded
scheduler GPU jobs. The login node only manages files and submits/queries jobs.
Each phase has logs, timeout and exit status; failed dependencies are cancelled.
Preparation and fit never overwrite any existing run. Every actual launch binds
source/config/input hashes. Raw rows, walls, traces and new research weights
remain private. Only reusable code, plans and aggregate results are published
to the research branch/PR8. No automatic main/Windows/model replacement.
The independent final confirmation grid remains unopened.
