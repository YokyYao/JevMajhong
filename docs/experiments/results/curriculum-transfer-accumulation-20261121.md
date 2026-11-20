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

## Matching later-course input contracts

`tools/prepare_curriculum_course_context.py` now prepares separate stage2/3 views for a possible context-v3 parent. It preserves every original row ID, group, label/soft target and loss-only tag/weight; only the same `public-riichi-context-v1` reach feature changes. The candidate manifests bind original files, transform/preparation code and input schema. They admit neither a parent nor a fit.

Stage2 has20,237 TRAIN/4,721 validation records:534/121 reach windows gain context and19,703/4,600 other windows remain unchanged after JSON parsing. Both the9,292 TRAIN and1,972 validation stage1 replay inputs/targets/groups match the context-aware parent views. The parent seal now checks full replay payload equality in both splits, not just the presence of TRAIN IDs; loss-only tags may change intentionally.

Stage3's informative kept pools remain22,959/5,905, with587/163 reach windows enriched and22,372/5,742 unchanged. The136/41 single-support quarantined rows remain quarantined and unchanged. These are still candidate files with the original split/exclusion semantics; they are not new optimal-action labels or proof of improved play. The original plain stage2/3 files and both running stage1 fits remain unchanged. All45 current curriculum checks pass, and actual stage2/3 preparations executed on allocated companion resources.

## Prospective research-parent admission amendment

A practical-v8 diagnostic on the same consumed107 validation reach windows gives TP19/FN18/TN31/FP39:51.35% acceptance,44.29% correct decline and47.82% balanced accuracy. Thus v8 itself fails the original requirement that both recalls reach60%. This measures the same flat pointer before the native discard guard. Some groups may have been used by v8 before the new curriculum reservation, so this is not a held-out generalization result, an expert-optimality certificate or an alternative to full-game evaluation.

This evidence shows that the original imitation gate cannot be the sole criterion for whether a fixed final is useful for **further research training**. It does not show that excessive riichi is harmless or that the current student matches v8. Preserve the original gate and its actual pass/fail result in the unchanged end audits. Keep per-class errors and targeted riichi work visible.

The [research-parent policy](../../../configs/training/curriculum-research-parent-v2.json) is an explicit prospective amendment, written **after development inspection**, not the original stage1 preregistration. It keeps sole final584 candidates and all execution, hash, group exclusion, input-contract and replay checks. Research continuation additionally requires:

- The original basic-discard retention limits.
- Both reach/discard actually chosen and balanced riichi agreement at least the practical-v8 diagnostic, with both recalls still reported. This does not certify the riichi issue as solved.
- The full64 development matches, not a16-game peak. Top-two/net/dealin must be no worse than the completed initial curriculum baseline:51.5625%,+18.75/match and18.6861% respectively. Practical v8 comparisons and paired intervals are reported separately; these development means do not prove equivalence or noninferiority.
- A new optimizer, unchanged input schema, and exact TRAIN **and validation** replay in the later course. No automatic release or relaxation of the unseen final comparison.

GPU registration bound the policy at2026-11-20T22:46:51 UTC, while stable-v2 had536 updates and context-v3 had149. Neither final584 evaluation had started. Policy SHA256 is `6ceff847b9ae2362ba079fe42372034c9f7d95f3f2b73b7921b061aa75b3c288`; the binding also freezes both registrations and the actual v8/initial-baseline diagnostic receipts. This prevents changing the research-admission criteria in response to either final584 result.

The policy binding is not a parent seal. The existing strict admission implementation remains strict; implementing and checking the separate amended research-admission route is still pending. No incomplete model, intermediate checkpoint or failing audit may bypass these conditions. Successful research continuation would still leave the full four-course goal, the riichi diagnosis and unseen v8-strength confirmation unfinished.
