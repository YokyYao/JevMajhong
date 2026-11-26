# Fixed cached riichi mechanisms diagnostic before changing a training recipe

2026-11-22. The basic fit315411 is still running; source-specific quality has
not passed. Validation76/152/228 already reveals a shift from frequent reach
to excessive refusal, with lower overall CE and some improved class ranking.
This is an explicitly retrospective diagnostic on consumed development data,
not a new confirmatory experiment or evidence identifying a causal defect.
Do not change the active608/four-epoch recipe or open the final test grid.

Freeze the following arithmetic before computing this diagnostic. Use only the
complete saved0/76/152/228 validation distributions on all3,266 unchanged menus,
with632 actual riichi windows, and the actual update1..228 source-count/CE logs.
No student/teacher inference, rollout, new label, weight update, threshold
search, action-selection change or parent admission is performed.

For legacy and teacher separately plus the pooled population, report class
probability AUC, reach-versus-best-discard logit-margin AUC, riichi-label Brier
score and ten fixed equal-width calibration bins/ECE. Preserve the stored
candidate order for actual argmax ties. The fixed0.5 full-class comparison uses
the same unmodified saved reach probability as the prior diagnostic, with no
new threshold selection. Calibration/decomposition arithmetic normalizes only
the tolerated floating-point sum error; report its maximum. These are
calibration/ranking against imitation labels, not win probability, optimality
or native/Windows policy strength.

Require exactly one reach alternative and otherwise ordinary discards for each
riichi window, exact menus/IDs and source tags, both label classes per source,
and finite nonzero relevant score mass. For these menus, normalized action CE
equals binary reach-versus-all-discard CE plus conditional chosen-discard CE
(the latter is zero for a reach label). Check this identity numerically; report
the components and probability/margin/discard-count distributions by actual
label class. Count disagreements between individual-action argmax and fixed
class0.5, including which labels they affect. This distinguishes the shape of
the action menu from poor separation/calibration without declaring either a
proven training cause or a recommended decoder.

From actual source counts times the registered per-example weights, reconstruct
every update objective and compute its effective riichi-positive/negative loss
weight, not an inferred gradient norm. Report fixed intervals1..76,77..152,
153..228 and the completed first epoch1..152. Show positive-share distribution,
counts below0.25/above0.75, aggregate shares and positive/negative weighted CE.
Verify the actual first complete epoch consumes each source count exactly once
and has50/50 positive/negative riichi loss mass. A globally balanced epoch can
have unequal individual updates. These descriptive relationships do not show
that batch imbalance caused a specific policy change.

Snapshot the append-only training and validation logs into a fresh private
output directory, retaining exact source/cache/checkpoint/config/code digests.
Do not bind the moving logs into the active run. Reproduce the stored raw
confusion counts at every point and verify all active/old completed source
bindings remain unchanged. New case-level margins/probabilities and derived
per-update weights remain private. Only source, this plan and aggregate evidence
may be published.

```bash
python -m unittest discover -s tests/core -p 'test_curriculum*.py'
python tools/audit_curriculum_riichi_mechanisms.py \
  --registered runs/curriculum-basic-repair-v1/fit-plan/registered.json \
  --config configs/diagnostics/curriculum-riichi-mechanisms-v1.json \
  --out runs/your-fresh-mechanisms-audit
```

All computation, hashing and tests run inside bounded scheduler GPU resources
with companion CPUs, node/GPU preflight, log, timeout and exit receipt. Do not
run this on login. The next actual complete-match point remains304;608/final64
and the final source-quality audit remain necessary before deciding whether a
separately registered RI objective/representation control is warranted.
