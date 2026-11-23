# Missing ordinary teacher decisions recovered and independently audited

2026-11-23, Asia/Dubai. Reusing the original256 Mortal trajectories adds37,068
previously uncaptured ordinary discard labels, with unchanged205 TRAIN/51
validation source-family splits. Collection and independent native/event-label/
encoder/split audits pass. **No new student training or performance gain is
claimed; the original calls388 fit continues unchanged.**

## Coverage gap and actual new data

The original basic collector samples plain own-discard slots1/4/8/12. Its
`seen` set is match-level: `(seat, discard_number)` is not reset at each new
hand. Thus later hands' same numbered slots are also absent, besides turns
outside that list. This was bounded source sampling, not evidence that all
ordinary trajectories had been taught. RI windows were collected separately.
The new [fixed input scope](../plans/curriculum-ordinary-completion-20261123.md)
captures every remaining single-actor own-draw menu containing only multiple
ordinary discards, with no meld and no player RI, outside all45,351 existing
teacher contexts. No chosen-action/result filter or yaku quota is used.

| New observed labels | TRAIN | Validation |
|---|---:|---:|
| Total |28,991|8,077|
| At old numbered slots1/4/8/12, but uncaptured contexts |8,104|2,344|
| Other turn numbers |20,887|5,733|
| Original physical source families |205|51|

Source families, not rows/turns, remain the split and independence unit. No new
source game or Mortal inference is generated. These are actual teacher
decisions in simulated source play, not human expert certification,37,068 new
winning hands or yaku/Win-PASS teaching. Earlier45,351 teacher records plus this
completion total82,419 records. The original actor reports82,440 network choices;
the21-count difference is not classified by this audit and must not be silently
filled with invented labels or used to claim complete action coverage.

## Labels come from explicit executed actions

An executed single-actor WaitAct discard is present in MJAI. The collector
checks the immediately following native/source dahai event, actor and tile,
then stores that directly observed label. Canonical PASS used to reconstruct
suppressed losing response intentions is only replay scaffolding: it never
becomes a negative chi/pon/kan label. Full recorded source events/final scores/
financial ledgers reproduce for all256 matches, with88,493 native batches and
503,316 bounded clone trials.

GPU317875 on ws-l2-016 completes0:0 in9m55s. The raw new output is fresh,
private and marked not admitted for training. Original records and the running
fit are not appended to, relabelled or overwritten.

The [independent audit](../plans/curriculum-ordinary-completion-audit-20261123.md),
GPU317902 on ws-l1-019, completes0:0 in3m41s. For every37,068 new record it
reconstructs the original physical current-hand prefix, including initial
hands/draws, and compares that exact event slice against the independent source
log. It verifies the public observation/menu/scope and derives the label from
the original explicit event offset, rather than trusting the producer's
selected-action receipt. Private physical histories are audit references only;
they do not enter model inputs.

All labels, family IDs and source seeds survive; normalized input matches the
ordinary serving feature contract. The fixed actual tokenizer admits every
record without truncation or legal-menu reorder. TRAIN maxima are1,178 packed/
581 state tokens; validation1,091/540. No old or cross-split duplicate normalized
input is found against available cleaned/calls/risk teacher references, and
all675 reserved old groups remain excluded. No policy/backbone is loaded.

## A remaining execution mismatch is explicitly retained

| Existing serving guard | TRAIN | Validation |
|---|---:|---:|
| Observed target allowed |28,573|7,984|
| Observed target blocked for shanten increase |418|93|

All511 blocked targets are retained unchanged. This proves an execution-
eligibility mismatch, not that every observed move is superior. Repeating
these labels cannot make the existing guard execute an excluded choice. The
already prepared defensive-scope proposal only broadens menus facing opponent
RI; every new completion case has no opponent RI, so that proposal does not
resolve these511 cases. Do not silently discard/relabel them or claim a
compatible training/serving contract is already admitted.

Next register and test the broader learned-choice contract, retain the fixed
RI-class boundary and legal-win behavior where applicable, and compare the
actual completed parent under changed selection before attributing any gain
to training. More ordinary source coverage is useful preparation, but it is
not a demonstrated cure for RI drift, weak daiminkan choices or the current
v8 performance gap. A subsequent data/mix/update-budget recipe and actual
native-game evaluation remain necessary.

## Provenance and limits

Collection sourcef65bae5e2caab2bb4f7d93cd9a0badcbfdbe9dcc and audit source
8564efd6ea83751b62d83021b4a8a24afa771a18 have exact pre-execution GitHub
readbacks for their respective implementation/config/entry/plan files. The
source replayer, original datasets and all224 active-fit dependencies remain
unchanged. Actual successful scheduler/shell exits are captured promptly.

All processing, native replay, tokenization and hashing uses finite scheduler
GPU allocations with companion CPUs, resource checks, progress logs, timeouts
and exits. No login compute, CPU-only workaround or unmanaged worker is used.
Only code/plans and this aggregate report are public; raw labels, observations,
source offsets/traces and new research weights stay private. Main, production
v8, Windows and the unopened final comparison are unchanged. The full
four-course/v8 goal is still incomplete.
