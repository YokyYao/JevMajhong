# Prospective progression objective and cadence, not a fit admission

The current calls fit uses max-discard versus RI binary training, while its
actual selector compares full RI probability with0.5. For two equal discard
logits, equality with their maximum corresponds to RI probability1/3, whereas
equality with their logsumexp corresponds to1/2. These are different objectives.
The actual final calls validation/native results and same-parameter route
comparison still require review before any parent/fit admission.

This new prospective helper uses a full-group RI binary CE plus conditional
discard CE on exact reach/dahai hard-target menus. Group coefficient2, conditional
coefficient1: lambda1 is exactly ordinary full-action CE and is not a novel
repair; lambda2 adds a real binary-class term. Pure ordinary/mixed menus keep
hard CE. Conditional MC retains masked soft-target CE, with zero gradient for
unobserved alternatives. This changes only the training objective, not the
fixed full-RI0.5 execution threshold, native legal menus or the one-pass model.

Proposed loss masses (before the additional binary term):55% complete existing
calls/basic/post-reach replay with its exact internal13-source proportions,
15% risk teacher,20% ordinary completion,10% old conditional MC split equally
across its four phases. This is19 sources, not raw-count proportional sampling.
Calculate fixed TRAIN example weights, report their cap and extra RI component,
and disclose the old MC conditional-hand/continuation limitations. Do not use
new VAL labels/outcomes to choose these arithmetic settings.

For the audited87,595 TRAIN rows, proposed accumulation73 gives1,200 optimizer
updates per full epoch, final update68 records. A511-record mixing block spans
seven updates; all19 sources fit once in each of172 blocks, including the214-
record last block. All original examples occur once per epoch, without copies.
It is impossible to put the194 distinct legacy RI-yes rows into every one of
1,200 updates without replacement. Therefore explicitly require full source
coverage per mixing block, not per optimizer update. Smaller updates are a
different optimization trajectory, not a proven remedy for decision variance.

Two epochs would mean2,400 updates and175,190 exposures. The provisional LoRA/
head rates3e-6/1.5e-5 are lower than current calls rates; neither matched total
learning-rate area nor equivalent Adam dynamics is claimed. Candidate would
be the final full exposure, with actual checkpoints/validation/game curves at
registered stage intervals. These numerical proposals are not a fit config.

The bounded GPU preflight checks the three numerical properties, all TRAIN
materialized menu/label/mask annotations, every source count/weight, and two
exact deterministic epoch orders with complete source exposure and recorded
macro/micro boundaries. For the first real example per source, verify finite
actual autograd components. Check all union audit/source and224 active-fit
dependencies unchanged. No validation labels are opened by this preflight,
no backbone/model is loaded, no optimizer is stepped, no game is played.

Public artifacts are only helper/config/plan/tests and aggregate results. Raw
data, order sequences, per-example diagnostics and new weights remain private.
The final fit recipe, actual admitted388 parent/route, exact tensor/probability
warm-start proof, curves and full untouched final-v8 comparison remain required.
