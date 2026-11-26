# Kan data lessons from the actual 3B step1250 snapshot

The running course continues to its registered 2500-update endpoint. This note
records a remaining capability gap; it neither vetoes the intermediate course
nor changes the pending placement lifecycle or the final confirmation criteria.

The [frozen-corpus audit](../results/curriculum-3B-1250-kan-labels-20261125.md)
uses all 91,247 training and 24,516 validation menus and cached predictions at
steps 0, 625 and 1250. Reconstructed original-kan confusion counts match the
actual validation journals. No new inference or simulations were performed.

The original teacher-kan training source has 281 legal daiminkan windows and
zero positive daiminkan labels. The legacy-kan source has 341 positive and 321
negative daiminkan labels. The MC-kan training source has 18 windows where its
largest labelled candidate favors daiminkan, 77 favoring another operation,
and 14 ties. These are target preferences, not expert judgments of profitable
play. Source-normalized loss masses also matter; raw menu counts alone do not
describe the effective mixture. Full coverage and masses are retained in the
aggregate CSV.

At step1250 the model selects daiminkan in 4/88 legacy positive validation
windows and 2/12 MC-preferred validation windows. The latter are small, consumed
diagnostic subsets. They support investigating coverage and action selection;
they do not establish the game value of taking those kans.

A separate, already completed contextual teacher collection had 554 source
cases, 537 network-labelled choices and two daiminkan choices. Both choices
were independently found in the saved teacher labels, submitted native batches
and actual native daiminkan suffix events. This rules out a completely missing
adapter path, not every contextual masking or policy bug. Both positive
examples belong to the original validation partition; none is reassigned to
training. The 395 training labels in that contextual collection contain no
positive daiminkan, so blindly appending that collection would add negatives.

If the final course still has this gap, the next data revision should:

1. Collect new, independent training families covering open and closed hands,
   discarders, useful replacement draws, dora exposure and opponent threats.
   Include realistic profitable-kan opportunities and legitimate declines.
2. Compare taking and declining in common sampled worlds. Keep uncertainty and
   soft targets for close outcomes; do not label every legal kan as mandatory or
   treat every teacher decline as an optimal-action proof.
3. Balance support within each kan type and label direction, while retaining the
   complete original replay. Report both example counts and effective objective
   mass, including support that the current teacher does not supply.
4. Keep source families in their original partitions. Evaluate the resulting
   choices alongside complete-East hand win, deal-in, top-two and net-point
   outcomes; a higher kan frequency alone is not success.

These are prospective data-design lessons. No additional fit or evaluation is
submitted by this note, no active dataset or bound source is edited, no raw
records are published and no production model is replaced.
