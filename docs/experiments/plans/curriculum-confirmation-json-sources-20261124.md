# JSON source evidence before the independent comparison

The independent final comparison remains unopened. A reserved seed written
inside a protected-range configuration is a declaration, while the same seed
inside a generated source record requires investigation. Blanket exemptions
for an entire configuration or registration file would hide that distinction.

`tools/curriculum_confirmation_json_sources_v1.py` adds a reusable, file-backed
check for an explicitly inventoried JSON subset. It parses nested values,
object keys and escaped strings, records JSON pointers and hashes, and rejects
duplicate keys and nonfinite numbers. JSONL streaming remains a separate check.

The only available declaration exemption is an exact protected-source-range
object with the original reserved start/count. Its file SHA-256 and specific
start pointer must be supplied. Other fields in that same file remain checked;
arbitrary score/data pointers cannot be exempted. A changed hash, malformed
range or declaration absent from the actual inventory is rejected.

The actual current 3A/frozen-v8 bound subset contains **307 JSON artifacts**.
The semantic scan matched their existing file identities and found exactly
**two protected-range declarations and zero unreviewed reserved occurrences**.
The teacher registration's 256 actual source families also exclude the range,
checked separately against its matching configuration. This supplements the
earlier file inventory; it does not establish complete future-candidate
provenance or prove historical non-use beyond the artifacts actually checked.

Nine new JSON-source checks plus the existing confirmation checks passed:
**26 checks total**, executed inside an approved GPU allocation. They test
data/declaration separation, escaped strings, altered hashes, duplicate keys,
nonfinite numbers and retained scope limitations. They generate no reserved
native games or model predictions.

Final admission still needs the actual qualified terminal model, complete
candidate/reference training lineages, full source-catalog and receipt-graph
coverage including future 3B/placement inputs, and a frozen final registration.
The new checker reports `confirmation_admitted=False`; it cannot supply those
missing requirements by itself. Raw source receipts remain private. This work
changes no active training inputs, weights, thresholds or advancement criteria.
