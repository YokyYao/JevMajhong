# Prospective unlabelled native kan-source pilot

Existing development evidence shows weak alignment on all three kan types.
Prepared new3B data supplies zero additional positive daiminkan labels. This
pilot constructs source opportunities for later paired action comparisons;
it does not force a teacher answer or change the current fit or 3B recipe.

Freeze192 plans: three kan types × four source motifs × eight templates ×
two physical world variants. Motifs are yakuhai/ordinary-tanyao ready shapes,
a closed-route control and a pair-heavy control. These describe construction,
not optimal route labels. All cases start early with equal scores and no
opponent riichi; they do not provide late defensive kan coverage.

Ankan sources use a genuine dealer draw of the fourth copy. Daiminkan
sources use a genuine first discard to an observer holding three copies.
Kakan sources use a legal pon, a scripted post-call discard, ordinary draws
and a later draw of the fourth copy. Full136-tile walls, legal physical
action traces and native public prefixes are retained privately. Scripted
prefix actions are never exported as teacher labels or negative examples.
Any prefix that would decline a legal win is quarantined without replacement.

Initial-hand and effective-decision shapes link suit/rank symmetry variants,
cross-operation controls and physical world variants into whole families
before native collection. Hash-based whole-family TRAIN/validation assignment
is fixed before labels or action values. Every emitted source must reproduce
its exact native prefix, public request and complete legal menu in a separate
replay. The target operation must actually be offered alongside alternatives.

Public model requests contain no construction motif, target instruction,
true wall or other players' hands. Hidden walls are source/replay artifacts,
not proposed teacher inputs or future sampling targets. Future labeling must
resample unseen information conditional on the public observation and compare
available actions using the same sampled worlds, retaining uncertainty and
sampling failures. No positive label quota is enforced by this generator.

This source-only pilot runs in a finite scheduler GPU allocation, including
an existing allocation's permitted companion CPU task. It loads no model and
produces no optimizer updates. It cannot admit SFT: full existing-corpus and
reserved-family overlap checks, actual conditional labels, coverage review,
normalization, loss allocation and a new training registration remain required.
Its output is separate from every current training/evaluation input; source
generation cannot promote a checkpoint or open the final reserved game grid.

Entry: `tools/generate_curriculum_kan_sources_v1.py`.
Recipe: `configs/diagnostics/curriculum-kan-sources-v1.json`.
Standalone scheduler template: `cluster/curriculum-kan-sources-v1.slurm`.
