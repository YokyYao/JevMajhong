# Alternative 3A launch handoff repair

The prepared augmented-data 2596-update trainer consumes
`launch-binding-private.json`. Its original activation wrote the registration
and launch gate but did not produce that separate file. No alternative student
had registered or started; the running 2472-update canonical replay is separate.

The V3 wrapper reuses the original activation function's code object in isolated
globals. It binds its additional source files into the actual registration,
checks the unchanged original registration and gate, creates the launch file
exclusively before submitting fit, and binds that file into the child queue.
Qualified-current and infrastructure-failure paths retain their original skip
behavior. Parent selection, losses, data, optimizer, update budget and execution
policy are unchanged. No control weights initialize the future fit.

Seven focused lifecycle tests passed in an allocated GPU job. They cover the
registration/gate/launch/submission order, queue source closure, both skip paths,
refusal of an existing launch file or changed prerequisites, and isolation from
the original module globals. This is metadata lifecycle evidence, not a real
alternative-model result. V2 had five passing tests and two fixture setup errors
caused by trying to exclusively create existing temporary files. Its source,
failed tests and execution logs remain preserved; V3 changes those two fixture
writes only.

After preparation passed, a finite replacement activation was submitted and
its actual script and unchanged parent-capture dependency were verified. Only
the verified own pending original activation was cancelled; its cancelled
terminal record was preserved. The replacement is pending on the current
canonical fit's capture. The running fit and its audit/capture are untouched.

The alternative remains conditional on a fully completed, independently audited
canonical final with the original failed strategy grade. A qualified canonical
final follows its existing 3B path; an infrastructure failure does not authorize
the alternative. No new student update, model publication, held final evaluation
or production replacement occurred in this repair. Scheduler records and raw
experiment data remain private.
