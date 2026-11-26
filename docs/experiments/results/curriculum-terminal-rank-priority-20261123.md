# Cached rank-priority comparison preserves pursuit and removes leading net-only ties

2026-11-23. The [prospective objective comparison](../plans/curriculum-terminal-rank-priority-20261123.md)
completes in GPU317859, ws-l1-002, exit0:0 in5s. Three independent numerical
checks pass. All original32-world scalar preferences and matched ordinary/
adaptive four-world values reconstruct from the saved records before any
comparison. No model is loaded, no new game/terminal arm is generated, and no
teacher label or student update is admitted.

The candidate ranks actions by expected terminal top-two frequency, then lower
expected terminal rank. Only if both tie and the **current public rank is3/4**
does bounded net break the tie. A current rank1/2 exact placement tie takes the
legal win. Integer counts/rank sums compare the first two components without
floating-point tie ambiguity. There is no positive-confidence gate or veto
against a lower fourth-place-escape rate; the numerical controls include a
fourth-place example where higher sampled top-two probability wins despite
much more fourth-place risk.

All16 previously fixed TRAIN views/eight families remain included. The two
four-world scenarios reuse indices0/1 from both saved blocks. Unique available
outcomes are1,024 ordinary terminal arms and128 changed outer arms; the128
ordinary subset outcomes are already part of the1,024 and are not new evidence.

| Same outcomes, changed objective | Scalar declines | Rank-priority declines | Changed choices |
|---|---:|---:|---:|
| Ordinary32, pursuit |6/8|6/8|0|
| Ordinary32, leading controls |1/8|0/8|1|
| Ordinary4, pursuit |4/8|4/8|0|
| Ordinary4, leading controls |2/8|0/8|2|
| Adaptive4, pursuit |4/8|4/8|0|
| Adaptive4, leading controls |5/8|1/8|4|

All changed leading choices were scalar top-two/rank ties broken only by net.
The remaining adaptive-four leading decline has a positive sampled top-two
difference, so the proposed objective does not impose an unconditional
"already second must accept" rule. All pursuit choices remain unchanged in
these caches. The ordinary32 pursuit declines include five sampled top-two
gains and one behind-net tiebreak. Both independent block choices agree in7/8
pursuit views and every leave-one-world choice agrees in6/8, unchanged from
the scalar objective. Ordinary32 leading controls now agree in all8/8 under
both diagnostics. That consistency is not a confidence certificate.

These results show that the proposed criterion expresses the intended
placement priority on this fixed dataset. They **do not demonstrate better
Mahjong policy strength**. In particular, adaptive-four outcomes were generated
by future replanning with the old scalar objective. Re-ranking those outcomes
does not execute a new future policy or provide32 adaptive worlds. Matched
ordinary/adaptive initial preferences can still differ after the change;
continuation assumptions remain material. The prior is uncalibrated and no
population/held-out policy claim follows from these eight families.

Original scalar code, outcomes, grades and the running calls recipe remain
unchanged. All224 active-fit dependencies retain their hashes. Actual
scheduler/shell exits are saved. Source6955df39c45158e2cfde941ca67c2262d5d6810a
has five exact pre-execution GitHub readbacks. Only the proposed helper, its
numerical tests, plan and this aggregate result are public; per-case preferences,
source IDs, allocations/outcomes and new weights remain private.

A future teacher needs a separately frozen native execution contract and
continuation check using its own objective, followed by reliable target
construction and actual student/native-game evaluation. This cached comparison
does not automatically select/admit that teacher. Main, production v8, Windows
and the unopened final comparison are unchanged. All local processing/tests/
hashing runs in a finite GPU allocation with companion CPUs, resource checks,
timeout/log/exit; no login compute is used. The full curriculum goal is incomplete.
