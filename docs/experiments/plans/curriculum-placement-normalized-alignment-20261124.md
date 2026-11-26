# Align terminal annotation to actual upstream normalization

The first full-data retained-placement preflight failed on a real terminal
row: upstream normalizes soft weights, while the old terminal annotator
requires bitwise equality with the raw weights. A tiny sum-rounding difference
therefore rejects a valid preserved teacher target. Keep that failed source,
registration, checked-count/error record and exit log unchanged.

The new v2 annotator derives an annotation-only question whose soft weights
exactly follow the pinned materializer's float conversion, option order, sum
and division. Require the encoded target to equal that normalized vector
exactly. Do not change the raw request, teacher labels, visible encoded input,
candidate keys/order, target mask, label, source or partition. Reject invalid
raw targets, altered encoded targets, partial RI and repeated annotation.
The old terminal loss then consumes the exact same encoded target; inherited
25-source replay still uses the exact retained3B loss and diagnostics.

Preserve the failed first v2 unit fixture and log: Python3.12's compensated
sum made ten identical0.1 values total exactly1, so that fixture did not
reproduce the intended nonunit rounding. The corrected fixture explicitly
requires a nonunit sum within the original tolerance; all rejection and
gradient assertions remain. The source backup is private and retained.

Five tests cover raw-input preservation, true rounding, full soft-RI values
and gradients, all25 inherited losses, zero gradient for untested options,
partial-RI rejection, incorrect normalization and annotation identity. Repeat
the complete91647-row available TRAIN proof in a distinct v2 directory,
report actual alignment count/max delta and48 representative gradients.
No missing targeted pool is imputed: nine collecting pools, full57-source
validation/weighting, trainer/auditor and actual qualified-parent training
remain required. No model, optimizer, game, active curve or final grid is
changed by this finite allocated-node software proof.
