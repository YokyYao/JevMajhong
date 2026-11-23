# Prepare reusable normalized3B teacher data with exact3A replay

Combine the existing audited87,595/23,688 progression replay with the new420/91
initial and3,232/737 continuation teacher labels. Preserve every replay row,
label, source, target mask and current authoritative file partition. Existing
native/token/mask audits are reused rather than rerun. Original136/41 singleton
quarantine rows remain excluded. The current3A fit is untouched.

Normalize only the4,480 new teacher decisions through the pinned shared
instruction/public-state/call-feature pipeline. Preserve candidate order and
action type/tile/consumed identities, selected hard action, original ID and
family partition. Retain original file/line/source and scenario provenance in
private metadata. Assign six route-index source tags for later recipe control;
the route is not visible input, an action instruction or proof of optimality.
These are teacher decisions, not automatically trusted route-value labels.

Build outcome-independent duplicate handling over normalized public requests:
preserve old replay, quarantine new rows already represented there, collapse
same-choice new duplicates, quarantine conflicting new hard labels, and
quarantine complete new families with opposite-partition visible-input
collisions. Never move a heldout family or choose duplicates by winning outcome.
Preserve quarantined payloads/reasons privately and report their actual counts.

Encode only admitted new records with the existing tokenizer and actual packed
limits. Require no truncation or label/menu reordering. No model or backbone is
loaded. The merged outputs retain original nineteen sources plus six route
sources, with per-source/route/phase and unique-family counts. All dataset
hashes, source manifests and source-family assignments remain versioned.

No loss mixture, optimizer budget, parent candidate or3B fit is admitted here.
Choose those after actual3A results and source-quality review, then bind this
data version in a separate recipe. Source outcome/teacher agreement and targeted
case frequencies do not establish student mastery or natural-game strength.

Run one bounded15-minute scheduler GPU job with companion CPUs/logs/exits;
no login processing or CPU-only workaround. Keep raw merged data/quarantine
private; public outputs contain code, plan and aggregate summary only.
