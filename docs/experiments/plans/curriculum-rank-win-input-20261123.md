# Stage4 public-win input and one-model execution contract

This candidate contract supports the new rank-aware win targets without
changing active3A/3B training, frozen v8, production or Windows.

Ordinary requests use the exact original normalization and complete-legal RI
policy. The new state field is accepted only for one legal win in scheduled
East4/extensions and is strictly limited to public win kind, source seat,
source type and tile. Hidden hands, walls, future scores, source IDs and target
diagnostics are not accepted as features. Public-source schema remains v1;
the normalization/instruction contract has its own explicit version.

Only these late win requests receive a terminal-top-two instruction in place
of the ordinary hand-win-first instruction. Training and inference share the
same projection. Preserve labels, targets and source/family metadata outside
the visible request. The existing raw candidate-label corpus remains immutable
and needs a separate normalized data output before fitting.

For a late ron, the current candidate comparison is legal hora versus native
PASS. For tsumo it is hora versus the current SAME model's most preferred
ordinary guarded legal discard. One shared LoRA and pointer head produces the
full probability menu once; no second model or residual network is added.
An exact pair tie takes the legal win regardless of menu order. Early wins,
single actions and current robbed-kan wins keep their prior shortcuts.
Non-win decisions retain the parent's complete-legal/fixed RI class route.

Teacher tsumo alternatives were selected by frozen v8. The student's future
best ordinary discard can differ; the two-arm supervision is an approximation
to continuing with that ordinary policy, not labels for every legal action.
Actual closed-loop native evaluation is required before treating this as a
strategy improvement. The one-call interface check cannot establish that the
untrained v8 can use the new instruction effectively.

GPU verification restores every263 selected native source view, executes
scripted win and continue selections with exactly one model-client call,
checks all strict tokenizer encodings and train/serve visible equality,
checks ordinary normalization on all24516 current validation rows, preserves
an actual early legal-win shortcut, and runs only four conditional plus two
paired ordinary real-model forward probes. These are interface evidence, not
new strength games or student validation curves.

Keep raw requests, traces and outcomes private; publish reusable code and
aggregate verification only. All tests, encoding, hashing and inference use a
finite scheduler GPU job with logs, device checks, timeout and exit status.
No main/Windows replacement, student update or final confirmation seed use.
Non-win terminal action-value targets remain a separate outstanding task.

After both label merge and this interface verification succeed, a separate
GPU data job preserves all91247/24516 original3B replay row objects and appends
the normalized rank-win targets in their original splits. It verifies all new
encodings, visible train/serve equality and unchanged source labels/targets/
family metadata. New outputs have their own hashes; original inputs stay
immutable. The receipt requires masked two-arm target loss and does not admit
a parent, optimizer, data mix or student fit. Candidate-label uncertainty and
quarantine counts remain explicit. This data step cannot start on an interface
failure. Remaining label shards run after that short check ends regardless of
its outcome, so independent data production can continue.
