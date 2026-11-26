# Automatically continue from completed targeted labels into Stage4

The actual3B fit and two targeted-world collectors continue on their current
GPU allocations. The fourth fixed shard waits for the user's normal GPU
quota. No collector is restarted or given fewer worlds or action arms.
Existing complete cases/worlds are reused by the already registered finite
recovery jobs; explicit sampling quarantines remain quarantines.

Queue the existing full-data producers after all final shard-capture jobs:

1. `curriculum_targeted_win_resumed_pipeline_v3.py --phase merge`, with four
   complete shard folders in original order and their genuine terminal
   archives. All108 original views must be accounted for, including explicit
   quarantines. A live/partial shard cannot be merged as complete data.
2. The same producer's `--phase prepare`, preserving original TR/VAL groups,
   all57 positive source pools and actual full projection provenance.
3. `prepare_curriculum_recovered_exposures_v1.py --phase mix`, then
   `--phase exposures`, preserving exact source masses, unchanged validation
   and the two count-derived epoch orders.
4. Only after exposures succeed and the actual3B capture is preserved,
   `launch_curriculum_placement_early_v1.py --phase submit` may admit the
   genuine completed parent and create the finite Stage4 phase graph.

Each data phase has a one-hour GPU allocation, four companion CPUs,48GB
memory, a55-minute command limit and an exit receipt. The final submission
phase uses a five-minute GPU allocation. Bind real scheduler job identities
while held, release dependents first, cancel own jobs if submission fails,
and use successful-completion dependencies with invalid-dependency cleanup.
There is no detached background poller and no login-node computation.

Queued dependencies do not imply that data has been merged or Stage4 has
started. Any incomplete actual shard or failed phase prevents subsequent
training. Independent frozen-v8 confirmation remains unopened by this graph.
