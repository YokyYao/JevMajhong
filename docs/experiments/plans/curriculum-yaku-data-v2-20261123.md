# Yaku data v2: correct the actual encoder metadata interface

Preparation318689 fails with KeyError('questions') because packed encodings
contain ids/segments/labels/option positions, while question keys and labels
belong to the materialized record. No teacher recollection, source change or
student update occurs. Keep v1 source/config/output and actual failure intact.

V2 uses the original materialized record for option-key/label alignment and
checks the actual packed labels and option count separately. Reuse the same
replay and all4,480 new teacher labels, normalization, family splits and
outcome-independent quarantine rules from the [original plan](curriculum-yaku-data-20261123.md).
No requirement is removed or label rewritten. Use new source/config/Slurm and
the data-v2 output directory; bind v1 failure and source as provenance.

This still produces reusable data only. Reuse old replay audits, encode only
new normalized cases, and do not load a backbone or alter the running3A.
Use one finite15-minute scheduler GPU allocation with companion CPUs/logs/exits,
without login processing or unmanaged workers. Publish code/plan/aggregates;
raw rows, quarantine and source histories remain private.
