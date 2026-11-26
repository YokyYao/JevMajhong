# Prospective RI retention anchor

This prepares one numerical objective; it does not change the continuing2472-update
trial, register a new student or qualify a later course. If the continuing trial
passes its original final criteria, follow its existing3B route. An intermediate
negative point cannot trigger this alternative automatically.

## Evidence and hypothesis

On the complete2353 original replay RI TRAIN examples, qualified initialization
to618 changes legacy balanced agreement62.4716% ->61.1867% and teacher agreement
67.3959% ->67.3422%. The cached632-menu VAL comparison has32 correct-to-wrong and19
wrong-to-correct transitions. No duplicated hard RI input/label conflict is found
in the full TRAIN inventory, and the two new hard teacher sources contain no
reach menus. The retained per-update sampler alone has not preserved RI behavior.
The earlier full-group/class-weight2 trial already failed; this is a different
prospective change, not a relabeling of that control.

Test whether an explicit probability-retention term reduces drift while new
examples teach useful behavior. Use the frozen qualified calls388 student as
the retention teacher, solely on TRAIN menus. This parent is not an optimal
teacher; anchoring it can also preserve mistakes or inhibit beneficial change.
Only actual source-quality and same-rule native outcomes can decide usefulness.

## Fixed numerical term

For each of the four basic-replay RI source classes, let `q` be the frozen
initialization's probability of reach against **all** discard mass. Let `p`
be the updated student's corresponding probability. Add

`4 * KL(Bernoulli(q) || Bernoulli(p))`

to that example's existing retained RI objective, before its registered source
weight. Strength4 is fixed prospectively, not searched against VAL. Keep the
original max-discard/conditional loss as its own reported component. All fifteen
other source objectives, including ordinary decisions, calls and masked MC,
remain numerically unchanged in this isolated helper.

Cache the parent probabilities offline, bound to the actual parent checkpoint,
complete TRAIN file, distinct IDs, visible inputs, original source/label and
complete action menus. Attach targets only as private loss metadata after the
pinned materializer determines candidate order. No VAL/test probability becomes
a training target. The numerical helper cannot establish provenance by itself;
the future registration must verify the complete source/checkpoint binding.

There is one student forward during training; the anchor uses its existing
candidate logits. Deployment still needs one ordinary student inference and
the same fixed RI0.5 selection rule. No retention teacher or correction network
is loaded during play. Candidate permutation must preserve the loss/gradient;
at the exact parent distribution the extra value and gradient must be zero.

## Current preparation and unfinished work

Five tests passed:zero value/gradient at parent, corrective gradients in both
directions, exact non-RI retained gradients, candidate permutation, and rejection
of incomplete/mismatched targets or changed visible inputs. A separate bounded
GPU step checked every2353 actual cached parent TRAIN distribution through the
pinned materializer; all satisfy the zero-anchor property. It loaded no backbone,
created no optimizer and made zero student updates. These are numerical checks,
not training observations or evidence of improved strength.

Before any future fit:complete and audit the independent256-family supplement;
freeze the exact combined TRAIN/VAL files, source mixture and full update budget;
generate/cache the qualified parent's probabilities for **every** final RI TRAIN
menu; integrate the weighted loss/observer into the actual trainer; and complete
a real full-model two-update optimizer/save/reload control in a separate output.
Record actual anchor KL, original objective, total optimized loss and the unchanged
canonical validation loss separately. A lower total training loss is insufficient.

Only the separately registered full budget may qualify; retain the existing
per-source RI, call/kan, risk and full-East native requirements. Log real loss,
reward, hand-win, top-two and net-point observations at fixed milestones. Keep
the reserved confirmation grid unopened until the final complete candidate is
sealed. This preparation does not establish v8 parity, replace the practical
model, merge main or publish private targets/trajectories.
