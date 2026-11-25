# Record actual 3B training and match-outcome curves

The active early-3B course remains a registered 2,500-update run. Reporting does
not change its data, optimizer, checkpoint choice, stage progression or policy.

`tools/plot_curriculum_development_report_v1.py` draws actual batch objective,
objective EMA (alpha 0.1), validation objective, match reward, hand-win rate,
top-two rate and net points. Match reward means terminal net points divided by
8,000; it is an evaluation outcome, not an RL optimizer reward trace.

The plot consumes the completed, source-bound report made by
`tools/summarize_curriculum_curve16_v2.py`. That reporter reconstructs native
financial ledgers and projects the first two preregistered development seed
groups from the complete 64-match endpoint. Thus each outcome curve uses the
same 16 complete East matches. Full 64-match endpoint aggregates remain in a
separate table. Two consumed seed groups do not establish independent strength
or superiority to frozen v8. Lines connect observations; no intermediate
outcomes are fabricated.

Actual execution on an allocated GPU node exported the 0/625/1,250-step curves
and all 1,250 original optimizer-loss observations successfully. PNG, SVG, PDF,
aggregate training-loss CSV and private source receipts were retained. These
are development artifacts, not the result of final confirmation.

`cluster/curriculum-early3B-final-report-v1.slurm` is queued after successful
completion of the actual fit's independent execution audit. It has one GPU,
one companion CPU, 8 GB memory and a 20-minute limit, with requeue disabled and
invalid dependencies cancelled. It verifies a prebound source checksum list,
exports completed points 0/625/1,250/1,875/2,500 and then draws the final curves.
It generates no games, loads no model and performs no optimizer updates.
If an input is missing or inconsistent, it fails visibly and retains logs and
its exit receipt; it does not fill in a result or overwrite partial evidence.

Raw matches, journals, paths and receipts remain private under `runs/` and
`logs/`. Code, aggregate metrics and figures may be published. The current
queue is not evidence that the final report, Stage 4 or reserved confirmation
has completed.
