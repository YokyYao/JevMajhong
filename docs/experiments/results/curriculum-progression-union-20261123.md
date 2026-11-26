# Risk and whole-hand progression corpus independently audited

2026-11-23. The next-course corpus now contains87,595 TRAIN and23,688
validation decisions. All111,283 row objects are independently compared to
their exact original sources and encoded through the actual pinned tokenizer.
The audit passes. **This is data readiness, not a new training result or
evidence of improved game strength.** Loss mix, update cadence and parent/fit
admission remain unregistered.

| Original records retained without relabelling | TRAIN | Validation |
|---|---:|---:|
| Current complete basic/calls/post-reach replay |49,022|12,479|
| Mortal closed-hand discards facing opponent RI |6,860|1,948|
| Missing ordinary Mortal decisions across later turns/hands |28,991|8,077|
| Informative old conditional simulation targets |2,722|1,184|
| Total |87,595|23,688|
| Whole-game groups in the union |2,960|930|

The Mortal records reuse the same205/51 source families and256 trajectories;
they are not additional independent matches. The simulation rows are old
hand-horizon conditional own-win/speed/dealin targets, not terminal-placement
or new Mortal labels. Source families stay the split/independence unit.

## Preserve labels, masks and held-out partitions

Observed targets retain their hard labels. All3,906 informative simulated
targets retain their existing soft probabilities and served support; unobserved
legal alternatives remain masked. The136 TRAIN/41 validation singleton-support
simulation rows stay in a separate original-source quarantine, because a
one-option masked softmax has zero loss/gradient. No confidence, reward-sign
or successful-outcome filtering occurs. The full original legacy validation
remains an explicitly identified frozen diagnostic reference.

Two preparatory software assumptions failed and remain recorded: v1/GPU317986
fails1:0 in1m27s on an absent historical row split; v2/317999 fails1:0 in50s
on a supplied legacy origin marker. GPU318019 completes0:0 in6s and inventories
the actual metadata. Current calls validation contains4,401 inherited TRAIN
origin markers; MC validation contains745 such markers and219 absent markers,
while MC TRAIN has501 absent markers. All Mortal family split markers agree
with their current files. Stricter curriculum holdout previously moved whole
families to validation while preserving original producer metadata.

The [v3 preparation](../plans/curriculum-progression-union-v3-20261123.md)
therefore uses each current hash-bound source file and whole-family exclusions
as the partition authority. It preserves every original object, including stale
or absent metadata, and never moves validation records into training. Explicit
Mortal family splits must still agree. Failed sources/tests/partial outputs/
logs/exits remain intact; v3 uses new paths and a regression for this distinction.

## Actual independent evidence

V3 preparation GPU318030 on ws-l2-014 completes0:0 in55s; four focused checks
pass. The [independent audit](../plans/curriculum-progression-union-audit-20261123.md),
GPU318056 on the same node, completes0:0 in2m55s. Actual scheduler terminal
records and shell exit receipts are captured for both.

The audit derives MC membership independently from original clean stage3 minus
stage2, checks exact replay ancestry, compares every merged row in exact source
order and verifies all structural quarantine rows. It materializes every label,
menu, private source tag and masked probability vector through the actual
serving format. All111,283 records tokenize without truncation or menu reorder.
Maximum packed/state lengths are1,602/637 TRAIN and1,412/618 validation. Eight
actual autograd probes cover all four MC phases in both splits; unobserved
alternatives have exactly zero target gradient.

Same-split duplicate public inputs are0 in both partitions. Opposite-split
record IDs, whole-game groups and public inputs have zero overlap; all675
original reserved groups remain excluded from TRAIN. The old TRAIN paired-world
evidence is hash-rechecked without opening validation rollout payloads. No new
world, policy inference or backbone load occurs. All224 active-fit dependencies
remain unchanged. Preparation/audit source6b76c1c1dfa944f56d28fa92d823200c4af4cfae
has exact pre-execution readbacks for all eight relevant files.

## Remaining training work

Raw source counts do not define the next loss mixture. Preserve specific RI
accept/decline diagnostics and five call-operation results when selecting the
actual mix; increased ordinary coverage alone does not establish a cure for
RI drift or weak daiminkan selection. The [complete-legal interface preflight](
curriculum-complete-legal-class-20261123.md) permits previously blocked teacher
choices while keeping the fixed RI class boundary. It does not prove the
student chooses those actions well.

The current calls run has completed388 real optimizer updates and98,044 record
exposures, with final validation/native callbacks and independent full audit
still pending at this report. The [same-parameter native comparison](
../plans/curriculum-complete-legal-class-native-20261123.md) is preregistered for
final388 and queued after that audit: reproduce16 unchanged-route games, then
compare64 proposed-route games against the audited same-grid cache. It has
zero training updates and preserves the original quality grade. Final actual
loss/reward/win/rank curve export is also queued after the audit, using the
unchanged verified exporter. No final performance value is claimed here.

Only after reviewing those actual results should the next loss mix, complete
epoch/update budget, exact tensor/probability warm start and sole candidate
rule be frozen for a risk/progression fit. No later fit, terminal teacher/student
admission, model replacement or Windows release is launched by this corpus
preparation. Final confirmation seeds remain unopened; v8 parity is unproven.

All computation uses bounded GPU scheduler jobs and companion CPUs with resource
checks, logs/timeouts/exits. No login computation or unmanaged background worker
is used. Only code/plans and aggregate results are public; raw records, labels,
traces, per-world evidence and new research weights stay private.
