# Reused-family advanced teacher data and replay preparation

2026-11-22. These are data/compatibility results while the fresh basic608 fit is still running, not a new calls model or evidence of v8 parity. Old failed calls/risk grades remain unchanged; no new parent is sealed and the final reserved test is unopened.

## Complete source and input checks

GPU315445 collects actual advanced choices with153 curriculum checks and exits0:0. All256 regenerated full physical matches equal the completed basic-course trajectories, including final scores. The original source log is reused by immutable reference. Thus these are more decision windows from the **same**205 TRAIN/51 validation families, not256 additional independent families.

GPU315449 independently reconstructs and normalizes all29,869 saved decisions and checks the actual tokenizer without truncation/action reorder. It exits0:0. All source labels, family assignments and public menus survive. New basic/legacy opposite-split inputs/groups/IDs and675 reserved groups are checked with zero cross-split observed-input overlap. Token maxima are1,487/1,412 packed and576/569 state, below4,096/8,192 limits. This does not recover unknown concealed-wall identities from historical public records.

| New teacher decision class | TRAIN | Validation |
|---|---:|---:|
| Chi/pon response |13,399|3,674|
| Post-meld discard/riichi |9,352|2,501|
| Kan-containing menu |729|214|
| Total |23,480|6,389|

Kan menus take precedence over chi/pon menus in this partition; action opportunities can overlap. Post-meld TRAIN/validation includes133/58 closed post-ankan choices, separately from9,219/2,443 open-hand choices. These actual closed cases are not falsely classified as open hands. Every eligible actual teacher choice is stored, including PASS and an intended call later overridden by another simultaneous action. Absence of a call in the outcome log is not used to invent a negative label.

| Operation | TRAIN offered | TRAIN selected | Validation offered | Validation selected |
|---|---:|---:|---:|---:|
| Chi |8,890|729|2,432|204|
| Pon |5,103|1,302|1,405|346|
| Ankan |299|56|88|15|
| Daiminkan |281|0|83|0|
| Kakan |149|47|43|14|

Not selected can mean PASS or another action. In TRAIN daiminkan offers,266 choices are PASS and15 are pon; validation has82 PASS and one chi. This cohort alone supplies **no positive daiminkan teaching example**. Preserve the legacy341 selected/321 not-selected TRAIN daiminkan cases rather than replacing them with this teacher's distribution. Neither opportunity counts nor matching the teacher proves when a kan is beneficial or whether the student has mastered it.

## Is the missing daiminkan a disabled action?

Private diagnostic315471 and the reusable public-entry verification315490 both complete with exit0:0. Their65 representative real-history fixtures come from the first64 completed source families, selecting offered operations, selected/other class and threat strata. They include18 reach,12 chi,18 pon,11 ankan,9 kakan and six daiminkan offers; counts overlap across fixtures.

The native and Mortal legal flags agree in all65 fixtures. In all six daiminkan cases, Mortal's explicit daiminkan flag and encoded kan action42 are enabled. Direct frozen-network Q inference selects PASS in all six, matching the stored actual teacher choice. This supports deliberate policy preference in these fixtures, rather than a disabled adapter action. It is **not** a full256-family eligibility sweep or proof that declining every daiminkan is optimal. No labels, student weights, live routing or game results change in this diagnostic. Per-case Q values/IDs/hands remain private.

## Prepared child replay, not an admitted fit

GPU315469 exits0:0 with159 curriculum checks, including source-specific parent rejection tests. The proposed [calls-repair data recipe](../plans/curriculum-calls-repair-data-20261122.md) retains every14,285/3,266 current basic row, adds legacy advanced rows once, and adds the new teacher advanced rows. It produces48,710 TRAIN/12,404 validation decisions from2,621/790 groups. All basic replay inputs/labels, actual advanced choices and675 reserved groups are preserved; decision/group overlap across splits is zero.

Its12 loss strata retain65% basic mass and35% advanced mass, with separate legacy/teacher and post-meld/call/kan categories. The count-only proposal selects accumulation254 and192 updates per full epoch; two proposed complete epochs would consume97,420 records. This is data preparation, not a registered optimizer or actual stage2 updates. Parent quality, exact warm start/loading, full training-entry checks, cadence and fixed candidate still require their own preflight/registration. Source-specific [parent admission](../plans/curriculum-basic-repair-parent-20261122.md) cannot be replaced by a pooled-success header or the older aggregate-only seal. The real parent is unfinished.

All compute uses bounded GPU jobs with companion CPUs, logs and exit receipts. The active basic model/input bindings and169 historical recovery bindings remain unchanged. Raw new hands, labels, trajectories, Q scores and weights stay private. Main, Windows and the practical v8 release are unchanged.

## Reproduction inside allocated GPU resources

After the [basic source preparation](curriculum-basic-teacher-data-20261122.md), use fresh outputs:

```bash
python tools/collect_curriculum_advanced_teacher.py \
  --config configs/training/curriculum-advanced-mortal-data-v1.json
python tools/audit_curriculum_advanced_teacher.py \
  --source runs/curriculum-advanced-teacher-data-v1/data \
  --out runs/curriculum-advanced-teacher-data-v1/audit-v1
python tools/audit_curriculum_mortal_masks.py \
  --basic runs/curriculum-basic-teacher-data-v1/data \
  --advanced runs/curriculum-advanced-teacher-data-v1/data \
  --out runs/curriculum-mortal-mask-audit-v1/public-entry-v1
python tools/build_curriculum_calls_repair.py \
  --basic runs/curriculum-basic-repair-v1/data \
  --advanced runs/curriculum-advanced-teacher-data-v1/audit-v1 \
  --out runs/curriculum-calls-repair-data-v1/data
```

The source collector verifies actual complete-trajectory equality and refuses silent replacement. `cluster/curriculum-advanced-teacher-data-v1.slurm` supplies separate bounded collection/audit phases. Use the pinned Mortal dependencies/runtime in the earlier reproduction guides. A new host must regenerate these private data from the recipe; raw historical labels are not published.
