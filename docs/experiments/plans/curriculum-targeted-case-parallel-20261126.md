# Parallel collection of the unchanged targeted cases

The remaining targeted validation collection was limited by one serial writer
for its last 27-view shard. The implementation now partitions only missing
source IDs between up to three workers. It retains the original 108 views,
source-family partitions, frozen v8 teacher, physical states, public-world
seeds, 32-world paired targets, future planner, legal menus and uncertainty.
It does not change the Stage4 training recipe or admit a smaller validation set.

## What changes

- Assignment is deterministic round-robin by the original missing-case order.
  No action advantage, target confidence or simulated outcome enters assignment.
- Each worker runs the original complete-world collector. The only AST change
  selects that worker's assigned source objects before the unchanged case loop.
- Original complete cases and explicit sampler quarantine are preserved.
  Atomic complete-world packets from ended parents are reused, while partial
  worlds do not become targets. Workers never share a live cache writer.
- A separate assembler requires every original view to have a complete case or
  its original explicit quarantine. It reconstructs the canonical complete
  27-view shard before the unchanged numerical full108 merge runs.
- The original downstream preparation, source mix, exposure accounting and
  Stage4 fit still consume the same canonical full-data outputs. Intermediate
  worker subsets are explicitly marked as incomplete shards.

The world-cache key retains the original numeric-producer identity so existing
packets can be reused. The new executor has a **separate bound implementation**
and actual scheduler provenance in its launch/completion records. These are
distinct identities; a parallel executor is not represented as an old batch job.

## Actual preparation and handoff

Six focused checks passed for disjoint coverage, object/order preservation and
complete preservation of the cached collector AST after reversing the sole
selection wrapper. Three additional lifecycle checks cover rejection of partial
lane input, rejection of an already running job as a new pending binding, and
held-before-handoff submissions.

On an allocated GPU, three already completed cases were restored in a different
order. All six original proposal blocks, including **96 public-world allocation
records**, matched their saved originals exactly with the frozen teacher. This
is empirical evidence for these fixtures, not a claim that every future rollout
has already been replayed or that model strength improved.

The serial writer was stopped through Slurm only after the new route was bound.
Its genuine cancellation terminal was preserved. The ended-parent snapshot
contained **10 complete cases**, leaving **17 cases assigned as 6 / 6 / 5**.
Previously written complete cases and caches were retained. The old pending
serial follow-ups and obsolete source-only observer were canceled after their
own pending state, ownership, command and zero resource allocation were checked.

## Finite resources and cleanup

Each case-worker attempt has one RTX 5000 Ada GPU, four companion CPUs, 48 GB
memory, an eight-hour limit, logs, an exit receipt and no automatic requeue.
Each lane has at most two queued attempts for its unchanged assigned population.
Actual scheduler terminal capture follows every worker and the full assembly.
The assembly and full merge have separate finite GPU jobs and cannot admit
incomplete source populations. Failed and canceled records remain preserved.

The queue allows up to three case workers plus the existing other shard's
collector. At handoff, the cluster enforced **three running jobs per user**:
two new workers ran alongside that original collector, while the third case
worker waited for capacity. This quota is respected. Long workers have lower
queue priority than their brief terminal-capture jobs so a newly freed slot can
preserve real exit evidence before it expires. No login-node computation,
CPU-only workaround, detached background process or automatic production update
is introduced.

## What remains unchanged and incomplete

The complete TRAIN inventory remains 91,718 independent records in 57 source
categories. The 80% retained-source / 20% terminal-source loss mix and the
count-derived two-epoch **2,470-step proposal** remain unchanged; actual full-data
preparation and registration are still required before training.

The new collection is not a trained checkpoint, a win-rate improvement or a
completed validation corpus. The actual full targeted-source observer needs a
new version supporting the parallel assembly lineage before final complete
source closure. Final Stage4 qualification and the unopened independent
frozen-v8 confirmation requirements remain intact.

Implementation:

- `tools/curriculum_targeted_case_partition_v1.py`
- `tools/curriculum_targeted_parallel_worker_v1.py`
- `tools/curriculum_targeted_parallel_assembly_v1.py`
- `tools/curriculum_targeted_parallel_capture_v1.py`
- `tools/launch_curriculum_targeted_parallel_v1.py`

Only code, this plan and aggregate observations are public. Original hands,
worlds, traces, research weights and private execution receipts are not published.
