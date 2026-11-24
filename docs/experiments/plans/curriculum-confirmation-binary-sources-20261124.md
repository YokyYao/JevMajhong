# Binary source evidence before the independent comparison

The independent final grid remains unopened. Raw paired-future training sources
include torch `.pt` artifacts; JSON/JSONL scans do not inspect their values.
This component checks an explicit inventory with original producer protocols
and source hashes. It cannot supply a missing complete candidate lineage.

Each raw file must retain its expected SHA256 and raw-rollout `rows` schema.
The restricted weights-only loader reads data into CPU memory inside a finite
GPU allocation. Weight checkpoint files are rejected as source records.
Native episode and counterfactual future seed fields receive explicit schema
checks. All stored nested keys/values, strings/bytes and numeric tensor elements
are checked for the reserved range2026300000–2026300031. Unknown object types or
unsupported tensor layouts fail rather than being silently skipped.

Tensor scanning uses bounded chunks. Numeric values in the reserved interval
are conservatively reported; nonfinite counts are recorded separately. Container
aliases/cycles are traversed once, which preserves absence detection but is not
an occurrence count of every possible alias path. Evidence concerns actual
stored values, not values that may have existed before a historical cast.

Every file receives a private receipt with hash, size, schema/counts, findings
and stability checks. Inventory completeness and producer provenance are
separate from value scanning. No blanket binary declaration exemption exists.
Any occurrence remains evidence requiring investigation, not automatic proof
that a final evaluation was run.

Five checks cover nested values/keys/boundaries, strings/bytes, integer/float/
complex tensors, unknown types, actual torch serialization, hash mismatches,
weight-file rejection and malformed future-seed fields. Actual data processing
requires a finite scheduler GPU step with logs and exit status. No model,
native reset, new rollout or final comparison is produced.

A clean explicit raw-MC subset cannot prove the entire final candidate/reference
ancestry, future3B/placement data, exact MC row selection/reward reproduction or
historical non-use beyond stored evidence. Those requirements remain separate;
the component always reports `confirmation_admitted=False`. Raw source records
and per-file seed findings stay private; publish only code and aggregates.
