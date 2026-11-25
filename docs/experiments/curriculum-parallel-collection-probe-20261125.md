# Targeted-label collection: small serial/parallel comparison

Three predetermined TRAIN worlds from a completed collection case were run
serially and with three owned spawned workers on one allocated GPU. The probe
kept every current action arm and inner plan unchanged. The complete native
trajectories, final scores, plans and counters matched exactly across the
stored reference, serial run and parallel run. No new training label or
student update was admitted by this probe.

| Timing | Seconds |
|---|---:|
|Serial execution, excluding initial model load|105.73|
|Three-worker execution, including worker model loads|102.86|
|Observed conservative speed ratio|1.028×|

This is about2.8% faster on three worlds, with no repeated timing sample or
interval. It is not evidence of a useful general throughput improvement over
the whole collector. Keep the current producers and their frozen identities
unchanged; do not introduce a new production collector solely on this result.

The finite GPU allocation and its terminal capture completed successfully.
Raw worlds, source cases, seeds, plans, trajectories, logs and detailed
process records remain private. This timing probe does not measure policy
strength, produce a new training result or open the final confirmation grid.
