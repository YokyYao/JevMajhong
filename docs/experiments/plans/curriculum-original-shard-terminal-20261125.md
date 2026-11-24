# Preserve the original label shard's actual terminal before recovery

The cluster reports `MinJobAge = 300 sec` and no usable accounting history.
Original shard00322030 is still running. Its previously queued12-hour recovery
322683 depends directly on the original terminal; a separate older15-minute
teacher job is also waiting for the two-job GPU quota. If recovery starts late,
the real original scheduler terminal could expire. An exit file alone does not
prove that a prior writer has terminated.

Bind a finite five-minute GPU capture job during the actual original writer.
It runs once afterany322030, queries that exact scheduler record, and verifies
the original command, node, array element and genuine phase-exit receipt using
the existing recovery terminal reader. It archives the real raw scheduler text
and exact exit receipt. It never polls, infers a missing terminal or fabricates
success. A timeout/failure of the original can be archived faithfully; the
capture itself succeeds only when the real evidence is available and valid.

Replace only the unstarted recovery submission with the same v3 attempt001,
same original prior and an explicit `--archive` argument, afterok the capture.
Preserve the old submission, cancellation and replacement receipts. Do not
overwrite its queue history or any collection output. Temporarily make the
pending daiminkan teacher depend afterany on the capture, so it cannot consume
the next freed GPU for15 minutes before archival. Preserve its downstream
capture and all resources. The running original collector and main training
remain untouched. The low-priority gradient probe remains separately queued.

This scheduling correction does not accelerate inference, change labels,
reduce32 worlds, alter source cohorts or start another student fit. Recovery
still validates every consumed case/world and the genuine archived terminal.
If capture cannot obtain real terminal evidence, recovery must stay stopped;
never bypass the writer-exit requirement.

Publish source, plan and scheduling summary only. Original scheduler/exit
archives and raw simulated data remain private.
