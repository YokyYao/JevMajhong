# Prospective larger-world stability diagnostic, TRAIN only

After the actual successful full256 source reuse and independent4,580-input
audit, test whether the unchanged public paired win-choice component produces
stable values on more independent TRAIN source families. No validation/final
seed is used to select this diagnostic cohort or change its kernel.

Choose two distinct source families per public bucket: third_gap3000/fourth_gap5000
crossed with ron/tsumo, at most one opponent RI, from the audited insufficient
small-win proxies. Sort IDs and select the first eligible unused family; eight
families must be distinct across all four buckets. The selection uses current
public scores/hand eligibility, not simulated future value or a desired label.
For each pursuit view include the same physical decision's hold_second view:
16 score views, eight source families. Fail if a fixed bucket is unavailable;
do not silently relax the cohort. Leading controls are not prelabelled accept.

Load frozen v8 once and use the unchanged public win planner/utility/prior.
Run two separately seeded16-world blocks per view,32 paired worlds total,
for exactly1,024 complete terminal arms across the16 views. Both current arms
share each world's future seed. Ron decline is native PASS; tsumo decline is
the same one ordinary guarded v8 discard. This does not value every legal
tsumo discard. Future inner wins remain rule-forced. The separately verified
observer-aware outer loop exists, but this diagnostic does not yet test its
continuation sensitivity or use it recursively.

Record pooled top-two/rank/net utility components, both block choices and
every leave-one-world choice, including disagreements and all-accept results.
No positive-CI gate or last-place-rate veto chooses labels. Stability metrics
are diagnostics, not confidence certification, target admission or population
policy win rates. Worlds share the same public decision;16 views are not16
independent families. The prior remains uncalibrated.

All raw cases, allocations, probabilities/decisions and full terminal events
stay private. Bind source/input hashes and actual completed input-audit exit;
check active-fit dependencies and teacher identity remain unchanged. Use a
finite1-hour1-GPU/4-CPU/48GB allocation with actual device/node checks, bounded
sampling and continuation, log progress per completed view, timeout and exits.
No student update, new source game, reward change, automatic teacher/target
admission, production change or final-test opening follows this pilot.
