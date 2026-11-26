# Public-context error strata and a fixed offline RI learnability probe

The running class-loss control's152 refusal recall is81/308 and its matched16
game result is poor. Do not modify or stop its frozen608-update experiment.
Diagnose existing TRAIN/validation decisions and cached76/152 predictions only.

Extract the existing label-independent public-context summary without adding
it to any model input. Before reading results, fix strata for opponent RI count
(0/1/2+), own tile-discard turn (0-5/6-11/12+), dealer status, score rank,
tenpai ukeire maximum (0-4/5-8/9+/unknown), and dora maximum (0/1/2+/unknown).
Report source/classes in TRAIN and validation and both models'76/152 source
confusions within those strata. Source counts and overlapping marginal strata
must not be described as independent samples or causal effects.

An offline logistic probe tests whether these labels have learnable signal in
an explicitly summarized public representation. Use the22 listed public scalar
features, each with a missingness flag; no source/label/group/seed/future result
is an input. Fit normalization on TRAIN only, keeping unit scale for constants.
Initialize one44-input linear binary classifier to zero and fit exactly500
full-batch AdamW updates (lr0.03, weight decay0.01, seed2026112305, float64 CUDA).
Use fixed RI-only source/class mass: legacy yes/no0.175 each, teacher yes/no0.325
each, derived from the current basic-course mass. No validation fit, threshold
tuning, alternate architecture, hyperparameter search or best-step selection.
Evaluate final500 with logit>=0 only after fitting. Keep probe parameters,
normalization, per-case outputs and training logs private.

This probe's binary task and engineered summaries differ from the joint
discard/RI Qwen pointer task. Success would show some accessible predictive
signal under this representation, not prove teacher optimality, isolate a
serialization cause or establish policy strength. Failure would not prove the
labels unlearnable. No native policy games, base/LoRA/pointer updates, parent
seal or later-course admission follow from probe recall.

Run tests/extraction/probe inside a finite scheduler GPU job with companion
CPUs, resource checks, timeout/log/exit. Require the existing actual data and
152 audit bindings unchanged before and after. Preserve all decisions/labels/
splits and the unopened final grid. Publish code/plan and aggregate diagnostics
only; keep new raw records, per-case predictions and weights private. Reconsider
the next isolated main-training experiment only after the actual608 final.
