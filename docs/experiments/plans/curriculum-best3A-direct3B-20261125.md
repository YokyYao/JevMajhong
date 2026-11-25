# Finish this 3A course, then continue its best trained checkpoint into 3B

User instruction on 2026-11-25 supersedes the previous strict stage gate:
after the current training finishes, select the best weights and proceed
directly to 3B. Do not start another3A merely because its stage grade fails.
Preserve the original grade and historical failed/prepared branches unchanged.

Complete the current2472-update fit. Compare the four trained scheduled
checkpoints618/1236/1854/2472 on exactly the same16 development cells, using
top-two rate first, net points second and the earlier checkpoint for an exact
tie. Step0 is a baseline, not a trained candidate. Do not select by validation
loss, use additional seeds to rescue a candidate, or open final confirmation.
This selection uses consumed development data and is not independent evidence
of superiority to v8. RI retention and v8 comparisons are reported, not vetoes.

Start the existing prepared3B course from that actual saved adapter and head
with a fresh optimizer: unchanged91247 TRAIN/24516 VAL,25 source masses,
retained19-source objective plus six route/control losses,2500 updates over
two epochs, and existing loss/reward/win/top2/net curves. No new2-update
integration control or strict3A qualification job is required. Real loading,
full parent validation parity and step0 same16 replay remain part of the
trainer's normal initialization. Reuse the selected parent's existing16 games;
if it is an interim checkpoint, complete only the other48 development games
to provide the existing final64 comparator. Those extra games do not change
the choice of parent and use no reserved confirmation seed.

Submit finite GPU prepare/fit/audit/capture jobs after the current fit. Once
the replacement graph is bound, cancel only the verified pending strict
canonical bridge and augmented2596 fallback activation. Preserve the running
fit and its original audit/capture. Full3B starts after preparation succeeds;
the old stage strategy grade does not block it. All calculations remain in
scheduler GPU allocations with limits, logs and phase exit records. This
change does not authorize a production model/EXE replacement or raw data
publication.
