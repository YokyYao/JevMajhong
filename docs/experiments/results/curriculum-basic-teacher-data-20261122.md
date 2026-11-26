# Targeted basic-course data and registered repair

2026-11-22. The four-course/v8 objective is not achieved. This report establishes a new data preparation and prospective fit, not a student playing-strength gain. Old plain584, negative context-v3 and failed calls/risk results remain unchanged. The final reserved comparison is unopened.

GPU collection315393 and independent audit315396 both completed with scheduler/exit0:0. All256 predeclared true-trajectory Mortal families complete, with205 TRAIN/51 validation families. Every seat/hand from a family has the same split. The collector forces legal wins and maps every other teacher action into the actual native menu, checking own-hand/event agreement. No hypothetical hidden-hand replacement, stale private prefix, invented score context or outcome filter is used. The source actor is the pinned Mortal298k; it provides action supervision, not initial student weights or an optimality certificate.

| New teacher decisions | TRAIN | Validation |
|---|---:|---:|
| Bounded early ordinary discards |3,106|769|
| Riichi selected |1,075|287|
| Riichi declined |812|238|
| Total |4,993|1,294|

All6,287 native public inputs and action menus reconstruct exactly from their physical source prefixes. Every label and whole-family split is retained. Normalization matches ordinary live UnifiedPolicyClient serving, and the actual pinned tokenizer admits every row with no truncation/action reorder. Maximum packed tokens are1,136/1,111 and state tokens565/552, below4,096/8,192 limits. Available earlier-course train/validation input fingerprints, IDs and groups are checked,675 reserved old groups remain excluded, and cross-split observed-input overlap is zero. This is not a claim to recover unknown historical concealed-wall identities from public decision records. New seed ranges were fixed before generation.

The256 complete source matches pass financial ledgers. Aggregate supported public-no-ura source-win yaku counts are diagnostics, not a prescribed mixture or learned yaku mastery. Special cases outside the narrow independent scorer, including kan hands, remain explicitly unsupported rather than assigned invented values. The actual decisions teach ordinary discard/riichi, not win/pass or complete scoring. New raw trajectories, teacher labels and detailed observations remain private.

## Mixed repair data and preflight

GPU315404 completed with exit0:0:150 curriculum checks,48 actual-row weighted-loss/gradient comparisons across all six sources/classes, and four complete sampler permutations. All169 completed risk-recovery bindings remain unchanged.

Preserve the entire legacy9,292/1,972 basic dataset and add all audited new teacher decisions. The result is14,285 TRAIN/3,266 validation records from775/214 groups. There are2,353 TRAIN riichi windows (legacy194/272 and teacher1,075/812), compared with466 previously. Validation has632 riichi windows, with legacy107 and teacher525 reported separately. No record is duplicated or relabelled; loss tags are outside the visible model input.

The [prospective fit plan](../plans/curriculum-basic-repair-fit-20261122.md) allocates65% loss mass to ordinary discards, half to each source. Riichi gets35%, half to accept/decline; within each riichi class35% belongs to legacy and65% to teacher. This prevents raw row counts from silently deciding the gradient mix. It does not make natural play frequencies50/50 or prove expert labels optimal.

The count-only accumulation rule selects94 records/update,152 updates/epoch and a sufficiently full final update. All six source/classes occur in every update without replacement. Four full epochs give608 actual updates and57,140 record exposures. Validation is every76 steps; matched16 complete-East diagnostics run at0/152/304/456, with64 at608 and the matching final16 subset kept for the curve. The sole candidate is final608. Loss is recorded each update; supervised reward is null, while real match reward/win/top-two/net/deal-in curves are separate.

The student uses fixed Qwen3.5-0.8B with fresh rank16 LoRA/256 pointer and the previous initialization seed/plain serving schema. It loads no old Mahjong policy or Mortal parameters. The data/mix/accumulation jointly change, so this is a combined recipe, not a single-factor ablation. Keep original aggregate quality controls plus unchanged legacy and new-teacher riichi controls separately. Source-specific parent admission, full later-course replay and the final v8 comparison remain required; no automatic later course or release follows this registration.

## Reproduction entry points

Use the pinned source/base setup in the [reproduction guide](../../../reproduction/README.md) and downloaded Mortal dependencies in the [source coverage guide](curriculum-stage4-input-coverage-20261121.md). Run all computation inside bounded GPU allocations. Output directories must be fresh; completed/failed prior artifacts are never overwritten.

```bash
python tools/collect_curriculum_basic_teacher.py \
  --config configs/training/curriculum-basic-mortal-data-v1.json
python tools/audit_curriculum_basic_teacher.py \
  --source runs/curriculum-basic-teacher-data-v1/data \
  --out runs/curriculum-basic-teacher-data-v1/audit-v1
python tools/build_curriculum_basic_repair.py \
  --teacher runs/curriculum-basic-teacher-data-v1/audit-v1 \
  --out runs/curriculum-basic-repair-v1/data
python tools/register_curriculum_basic_repair.py \
  --config configs/training/curriculum-basic-repair-v1.json \
  --base-path "$BASE_SNAPSHOT" --reference-path "$FROZEN_V8"
python tools/train_curriculum_basic_repair.py \
  --registered runs/curriculum-basic-repair-v1/fit-plan/registered.json
python tools/audit_curriculum_basic_repair.py \
  --registered runs/curriculum-basic-repair-v1/fit-plan/registered.json \
  --out runs/curriculum-basic-repair-v1/final-evidence-private.json
```

`cluster/curriculum-basic-teacher-data-v1.slurm` supports separate collection/audit phases. `cluster/curriculum-basic-repair.slurm` runs an already registered fit with a resource preflight and bounded timeout/exit record. A fresh host without the historical v8 reference cache sets `reference_evaluation_cache` to null in a versioned local config **before registration**, then computes the same reference grid rather than faking a cache. Local paths and bundle metadata can differ; fixed numerical configuration, actual input/model tensors and real metric trajectories are the reproducibility targets. This report does not promise bit-identical CUDA weights across different hardware.
