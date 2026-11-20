# Curriculum transfer and accumulation diagnostics

2026-11-21, Asia/Dubai. These changes prepare later courses while both stage1 fits keep their frozen data/source/configuration and fixed584-update budgets. No new course fit, parent seal or replacement production policy is created by these checks. The [full goal](../plans/curriculum-optimization-20261121.md) still requires the four courses and an unseen paired comparison with practical v8.

## A short final update changes loss composition

Stable-v2 uses9,292 records and accumulation64:145 complete groups plus a final12-record update each epoch. The pinned upstream trainer divides each group's summed loss by its actual record count, including the final12. The current balanced scheduler forces both riichi classes into that small group.

The actual end-of-epoch updates146/292/438 each contain9 plain and3 riichi records. With frozen weights0.84223884 plain/3.98798283 riichi, their normalized loss-coefficient share is61.215% riichi, versus the planned20% aggregate epoch mass. At146 the small group has2 accepts/1 decline; at292/438 it has1 accept/2 declines. Pre-clip gradient norms are82.864/24.060/20.924. These are observed coefficient compositions and total gradient norms, not measured per-source gradient contributions or proof that the short group causes validation oscillation. Adam state, clipping, examples and earlier updates also matter. Normalization is implemented as designed; this is a recipe concern rather than an arithmetic-loss bug.

`tools/audit_curriculum_accumulation.py` checks unchanged registered inputs and reads actual complete update logs. It does not load a model, change weights, fit a threshold or rerun games. Its count-only prospective rule chooses the largest accumulation between half the requested maximum and that maximum, requiring a final group at least75% of a full group and enough distinct rare examples for every update. For this stage1 dataset it proposes62, giving150 updates/epoch and a54-record tail. This candidate is **not trained or admitted**, and neither running fit is changed. It is not an improvement claim.

## Later-course scheduler without copied examples

For the prepared stage2 counts20,237 TRAIN and a requested maximum128, the same prospective rule proposes accumulation125:162 updates/epoch, with112 records in the final update. This avoids128's13-record final group. Accumulation does not have to be a power of two when the GPU microbatch remains1.

`tools/curriculum_course_order.py` constructs a complete without-replacement epoch with all six declared loss strata in every optimizer group: plain, riichi-selected, riichi-declined, open discard, call and kan. It first allocates a distinct example from each stratum to each group, then shuffles/distributes all remaining original records within declared capacities. It adds no rows and changes no label or epoch-wide loss weights. Each epoch records the real per-update counts/order digest. Individual operation types such as kakan remain separately audited coverage; the sampler does not promise every individual operation in every update.

Two diagnostic epoch permutations using the **real prepared data** each contain exactly20,237 records,162 update groups and a112-record final group; every group has all six strata. This is data/scheduler evidence only, not an actual training epoch. An eventual fit still needs an explicit accepted parent, fixed budget/configuration and retained-course/game evaluation.

## Continue the exact accepted adapter/head with a new optimizer

The pinned legacy parent verifier selects a conventional minimum-CE/registered research candidate. The curriculum instead names an audited fixed-budget final checkpoint with a stage-quality gate. `tools/curriculum_transfer.py` bridges that explicit seal into the existing warm-start machinery without editing the frozen research files or bypassing group/lineage/tensor-load checks.

The bridge requires the seal hash, exact selected checkpoint/model/base/source identities, all bound files and the actual completed stage metrics. It rejects a legacy minimum-CE fallback, an incomplete/unqualified seal, mismatched child datasets and optimizer resume. Context-aware parents also need the complete raw-public serving evidence, matching checkpoint contract and a child with the same input schema. The supplied context manager replaces only the parent-reader function for that training process, restoring it on exit; the pinned `bind_parent`, lineage/group checks and exact loaded adapter/head tensor checks remain in force.

An actual path incompatibility was found before child fitting: stage1 protocols bind two dataset entries with relative paths, while the pinned ancestry verifier looks up the absolute trainer paths. `canonical_protocol` verifies every original file hash, adds matching absolute aliases in memory, rejects conflicts and preserves the original entries/file. It does not rewrite parent provenance or relax hashes. The checked parent seal additionally names the exact child data-ready manifest.

`tools/check_curriculum_transfer.py` exercised the data/lineage part on real stable-v2 and stage2 artifacts. The pinned lineage/group checks pass:570 parent TRAIN groups,2,416 child TRAIN groups and739 child validation groups, with zero overlap. All9,292 mandatory stage1 replay rows retain the same label and model-visible request. Parent validation, child validation and675 reserved groups remain excluded from child training. The original parent protocol hash remains unchanged. No student is loaded, no parent is admitted and no optimizer update runs.

The positive **complete qualified-parent read and actual child warm start** still need a real final parent that passes its audit and stage criteria. The static/data checks above do not prove that this full path has executed. First-course completion, the later fits and final strength comparison remain outstanding.

Run all commands in an allocated GPU job with companion CPUs, logs and a timeout:

```bash
python -m unittest discover -s tests/core -p 'test_curriculum*py'
python tools/audit_curriculum_accumulation.py \
  --registered runs/curriculum-optimization-v2/fit-plan/registered.json \
  --out runs/curriculum-optimization-v2/accumulation-composition-private.json
python tools/check_curriculum_transfer.py \
  --registered runs/curriculum-optimization-v2/fit-plan/registered.json \
  --child-data-ready runs/curriculum-optimization-v2/stage2-data/data-ready.json \
  --out runs/curriculum-optimization-v2/transfer-data-preflight-private.json
```

Use fresh destinations; receipts are not overwritten. Public material contains code/tests and this aggregate summary. Original records, order receipts, gradients, predictions and unreleased weights stay private.
