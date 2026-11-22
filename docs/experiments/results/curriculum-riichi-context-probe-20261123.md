# RI errors: public-context signal exists, but Qwen transfer is unresolved

2026-11-23. A fixed offline classifier separates accept/decline labels better
than the current Qwen control's152 snapshot on the same consumed validation
RI windows. This is diagnostic evidence about the data and representation,
not a trained Mahjong policy, parent qualification or playing-strength gain.
The full frozen608-update Qwen experiment continues unchanged.

## Existing decisions and a fixed probe

Reuse all2,353 TRAIN and632 validation RI windows from the active14,285/3,266
basic dataset. Whole-source groups/IDs are disjoint across splits. TRAIN has
legacy194 accepts/272 declines and teacher1,075/812; validation has legacy37/70
and teacher287/238. No new hand, teacher action or label is generated.

The [registered diagnostic plan](../plans/curriculum-riichi-context-probe-20261123.md)
uses22 allowlisted public-context scalars plus missingness flags (44 inputs):
scores/placement gaps, dealer/turn/threat information, honba/sticks and summaries
of legal tenpai discards' ukeire/safety/value-honor/dora/red material. These are
public summaries, not complete yaku/han/fu values or future results. Source,
label, seed, group and future outcome are excluded from inputs. Normalization
uses TRAIN only; constants retain unit scale.

One zero-initialized linear binary classifier fits exactly500 full-batch CUDA
float64 AdamW updates, lr0.03/decay0.01/seed2026112305. RI-only source/class
mass is legacy yes/no0.175 each and teacher yes/no0.325 each. Evaluate final500
with fixed logit>=0; no validation fitting, threshold search, alternate model
or best-update selection occurs. Probe fitting and its reproduced run are
separate from Qwen optimizer steps and from the actual Mahjong reward curves.

| Consumed validation RI windows | Qwen class-control152 accept / decline | Offline final500 probe accept / decline |
|---|---:|---:|
| Legacy |33/37 (89.19%);19/70 (27.14%)|28/37 (75.68%);54/70 (77.14%)|
| Teacher |248/287 (86.41%);62/238 (26.05%)|199/287 (69.34%);174/238 (73.11%)|
| Pooled |281/324 (86.73%);81/308 (26.30%)|227/324 (70.06%);228/308 (74.03%)|

Pooled probe balanced recall is72.04% versus the Qwen snapshot's56.51%.
The probe's TRAIN balanced recall is74.15%. Its lower accept recall is an
explicit tradeoff, not a uniformly better decision. The probe cannot select
discard tiles, calls or wins, and has not played native matches. Its binary
task, engineered feature vector and dedicated loss differ from Qwen's joint
discard/RI pointer training; this is not a controlled architecture comparison.
No Qwen grade is replaced by the probe's numbers.

## Where the current152 errors occur

The strata were declared before extraction and use public tile counts, not
the earlier incorrect river-text-length counter. Selected marginal summaries
of actual Qwen152 type recalls are below. Denominators are labels within each
stratum; strata overlap and are not independent samples.

| Public stratum | Accept recall | Correct decline recall |
|---|---:|---:|
| No opponent RI |222/254 (87.40%)|31/171 (18.13%)|
| One opponent RI |55/64 (85.94%)|49/132 (37.12%)|
| Own prior discards0–5 |59/66 (89.39%)|1/26 (3.85%)|
| Own prior discards6–11 |199/225 (88.44%)|40/179 (22.35%)|
| Own prior discards12+ |23/33 (69.70%)|40/103 (38.83%)|
| Maximum tenpai ukeire0–4 |94/122 (77.05%)|72/255 (28.24%)|
| Maximum tenpai ukeire5–8 |173/187 (92.51%)|9/48 (18.75%)|

Refusal errors are not confined to reacting defensively to an opponent's RI:
140 of the171 no-opponent-RI decline labels are still predicted as reach.
Those source classes are present in TRAIN (legacy198/teacher487 declines), so
their mere absence is not the explanation. Early decline examples also exist
(legacy37/teacher90 TRAIN;8/18 validation), but25 of26 validation early declines
are missed. These counts do not prove the labels optimal, establish the cause
of game losses or show how much new data would help. Do not filter/relabel
these difficult cases or adjust active sampling after seeing validation.

## What this does and does not change

The larger mixed cohort contains predictive public signal for RI imitation;
it is unsupported to explain failure solely as too few labels or no visible
signal. Joint-task interference, optimization and access to the useful summary
remain hypotheses rather than established causes. More random source games
alone are not demonstrated to fix them by this probe.

The [earlier smaller-cohort diagnostic/context-v3 experiment](curriculum-riichi-context-20261121.md)
already found a successful offline probe and an unsuccessful full Qwen input-
enrichment control. Its final64 net mean is−332.81 versus plain+1,760.94 and
v8+4,290.63. The new probe changes data/features/budget from that older probe,
so its recall is not a controlled before/after comparison. Do not repeat the
old enrichment recipe or deploy a separate corrective classifier merely
because this new binary probe is good.

Retain the actual completed negative experiments and finish the current608
control. If its audited final still fails, use source-specific train/validation
class components and ordinary/RI interaction diagnostics to specify the next
isolated main-training change. Full policy games, later calls/risk/placement
courses and the untouched32-seed final comparison remain required. No source
gate is waived and no unfinished student is admitted to a later course.

## Execution and independent reproduction

GPU316023 on ws-l3-004 completes scheduler/exit0:0, passes225 curriculum checks,
extracts the actual inputs and fits the fixed final500 probe. Independent
GPU316039 on ws-l2-004 completes0:0 and checks all2,985 actual feature vectors
under changed labels, source and group/future metadata: every vector is
unchanged. Public turn/threat/dealer/rank summaries cross-check the raw public
state; TRAIN/validation groups and IDs remain disjoint. It refits the same500
updates from zero using TRAIN alone. Normalization is exactly reproduced and
maximum parameter, training-loss and final-logit errors are all0. Both split
profiles reproduce exactly. This establishes implementation reproducibility,
not independent population generalization.

All work runs inside finite GPU scheduler allocations with companion CPUs,
resource checks, timeouts/logs/actual exits. Active/completed bindings and main
Qwen/LoRA/pointer weights are unchanged; no native matches or policy inference
are added. Code, plan and aggregate evidence are public. Per-case context,
predictions, training details and probe parameters remain private. The live
model, Windows release and main branch are unchanged; the four-course goal
is still active and unachieved.

Published source27092fd2677fc914a55d2ab338b532ce5a852c2a passes GitHub
CI45082727644, both core and Windows desktop jobs. The four published source,
test, plan and initial aggregate-report contents are read back exactly at that
commit; this later report-only addition records the successful CI receipt.
