# Prepare RI-loss retention for a future calls course

The active fresh basic control changes only RI binary class loss. The existing
prepared calls trainer instead applies ordinary hard action CE to every basic
replay row. If the current control later qualifies, using that trainer without
an explicit decision would switch its learned RI objective back to the original
definition during retention. This is a compatibility risk, not an observed
forgetting result: the new basic control has not yet finished, and no repaired
calls child is admitted or trained.

Prepare a separate loss adapter for the unchanged12-source48,710/12,404 calls
dataset. Map only the six basic replay source tags temporarily into the active
six-source annotation/loss helper; preserve original source, label and key
order. Basic RI yes/no use exactly its max-discard binary class plus conditional
discard objective. Basic plain and all six advanced sources keep the existing
calls CE, including its dtype/autocast behavior. Advanced reach/kan alternatives
are deliberately not redefined by this proposal. Original hard CE and changed
objective remain separate diagnostics. No new head, model pass, decoder or
serving filter is added.

The adapter adds only private loss indices/marker after materialization. Verify
all61,114 actual prepared inputs and actual packed encoder fields unchanged,
without truncation or key/label reorder. Compare value/gradient on actual-row
CUDA logits against the old calls CE for unchanged sources and against the
active basic RI helper for basic RI. These deterministic synthetic-logit checks
test the loss implementation, not model inference, teacher optimality or playing
strength. Preserve every record/source/label/split and the existing675 reserved
groups; this preparation changes no dataset or weight mass. Reject new post-
reach source tags and soft targets rather than silently admitting a new mixture.

Run the unit/native-token/loss preflight in a finite scheduler GPU job, with
node/resource checks, timeout, logs and exit. Load tokenizer only, no language
model parameters or optimizer. Verify all current/completed source bindings
before and after; no changes to active RI control or existing calls entries.

This module and its preflight are not a registrar/trainer, qualified-parent
seal, loss-recipe admission or launch. Only after the basic608 control's actual
audited source quality and games decide the child recipe. Full calls training
still needs source-qualified parent loading, exact warm-start tensor/prediction/
game equality, separately registered complete epochs/exposures/learning rates,
retention controls, real curves and independent audit. Its actual class-loss
metric definitions must be explicit in both train and validation. The existing
calls recipe remains preserved, and312 new post-reach TRAIN rows are outside
this12-source proposal. The complete four-course/unopened-v8 goal remains.
