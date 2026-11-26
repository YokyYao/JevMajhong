"""Observe only newly inventoried JSON metadata; never admit final games.

Closed rollout payloads are reused, not rescanned. Other input types and new
reserved occurrences require a separate source observer or explicit review.
Execute only inside a finite scheduler GPU allocation.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_confirmation_json_sources_v1 import _object, _constant, _protected_range
from tools.curriculum_confirmation_contract_v1 import CRITERIA
from tools.curriculum_transfer import sha

NUMBERS = re.compile(r'(?<![0-9])[0-9]{10,20}(?![0-9])')
MAX_BYTES = 2 * 1024 * 1024


def decode(raw):
    return json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)


def findings(value):
    """Inspect decoded scalars and keys, including exponent-encoded numbers."""
    counts = Counter(); hits = []; stack = [('', value)]
    start = CRITERIA['seed_start']; groups = CRITERIA['seed_groups']; stride = 1 << 32

    def number(pointer, kind, value):
        if start <= value < start + groups:
            hits.append(dict(pointer=pointer, kind=kind, encoding='logical', value=value))
        delta = value - start * stride
        if 0 <= delta < groups * stride and delta % stride < 100000:
            hits.append(dict(pointer=pointer, kind=kind, encoding='native_spaced_hand', value=value))

    while stack:
        pointer, item = stack.pop(); counts['values'] += 1
        if item is None or type(item) is bool:
            continue
        if type(item) is int:
            counts['integers'] += 1; number(pointer, 'int', item)
        elif type(item) is float:
            counts['floats'] += 1
            if not math.isfinite(item):
                raise ValueError('Nonfinite decoded metadata cannot support evidence')
            if item.is_integer():
                number(pointer, 'float', int(item))
        elif isinstance(item, str):
            counts['strings'] += 1
            for match in NUMBERS.finditer(item):
                number(pointer, 'str', int(match[0]))
        elif isinstance(item, list):
            stack.extend((pointer + '/' + str(i), child) for i, child in enumerate(item))
        elif isinstance(item, dict):
            for key, child in item.items():
                if not isinstance(key, str):
                    raise ValueError('JSON metadata requires string keys')
                where = pointer + '/' + key.replace('~', '~0').replace('/', '~1')
                stack.append((where + '/@key', key)); stack.append((where, child))
        else:
            raise ValueError('Unexpected decoded JSON metadata type')
    return dict(counts), hits


def observe_file(path, expected):
    path = Path(path).resolve(); before = path.stat()
    if path.suffix != '.json' or before.st_size > MAX_BYTES:
        raise ValueError('Use the dedicated observer for non-metadata or large inputs')
    raw = path.read_bytes(); actual = hashlib.sha256(raw).hexdigest()
    if actual != expected:
        raise ValueError('Inventoried metadata identity changed before observation')
    value = decode(raw); counts, hits = findings(value); reviewed = []; unreviewed = []
    for hit in hits:
        if hit['encoding'] == 'logical' and hit['kind'] == 'int' and hit['value'] == CRITERIA['seed_start']:
            try:
                _protected_range(value, hit['pointer'], CRITERIA['seed_start'], CRITERIA['seed_groups'])
            except (ValueError, KeyError, TypeError, IndexError):
                unreviewed.append(hit)
            else:
                reviewed.append(hit)
        else:
            unreviewed.append(hit)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns) or len(raw) != before.st_size:
        raise ValueError('Metadata changed during observation')
    return dict(path=str(path), sha256=actual,
        role='actual_new_bounded_JSON_metadata_values_observed', counts=counts,
        stable_during_read=True, bytes_observed=len(raw),
        reviewed_exact_protected_range_declarations=reviewed,
        unreviewed_reserved_occurrences=unreviewed)


def observe(inventory, out):
    allocation = resource(); started = time.monotonic()
    inventory = Path(inventory).resolve(); out = Path(out).resolve()
    if out.exists():
        raise ValueError('Never overwrite complete or partial source observations')
    binding_path = inventory / 'completion-binding-private.json'
    binding = decode(binding_path.read_bytes())['files']
    files = {str(binding_path): sha(binding_path), str(Path(__file__).resolve()): sha(__file__)}

    def checked(name):
        path = inventory / name; actual = sha(path)
        if binding.get(str(path)) != actual:
            raise ValueError('Completed delta inventory evidence changed')
        files[str(path)] = actual
        return decode(path.read_bytes())

    summary = checked('summary-private.json')
    items = checked('additional-artifacts-pending-observation-private.json')
    issues = checked('declaration-issues-private.json')
    if (summary['status'] != 'actual_source_delta_inventoried_not_complete_source_proof'
            or summary['declaration_issues'] or issues or summary['unread_large_artifacts']
            or summary['additional_artifacts'] != len(items)):
        raise ValueError('An actual bounded, issue-free completed metadata inventory is required')
    observations = []
    for name, item in sorted(items.items()):
        if (name != str(Path(name).resolve()) or item['path'] != name
                or item['suffix'] != '.json' or item['size_bytes'] > MAX_BYTES
                or item['reused_completed_source_observation']
                or not item['actual_bytes_hashed_in_this_inventory']
                or not item['typed_source_observation_pending']
                or binding.get(name) != item['sha256']):
            raise ValueError('Every new input requires its actual inventory-bound metadata identity')
        row = observe_file(name, item['sha256'])
        if row['unreviewed_reserved_occurrences']:
            raise ValueError('Unreviewed metadata seed occurrence requires context review: ' + name)
        observations.append(row); files[name] = row['sha256']
    out.mkdir(parents=True, exist_ok=False)
    write(out / 'per-source-observations-private.json', observations)
    result = dict(status='actual_new_bounded_metadata_delta_values_observed_not_final_source_proof',
        actual_metadata_files=len(observations),
        actual_bytes_observed=sum(row['bytes_observed'] for row in observations),
        reviewed_exact_protected_range_declarations=sum(len(row['reviewed_exact_protected_range_declarations']) for row in observations),
        reserved_source_data_hits=0, unreviewed_artifacts=[],
        old_closed_source_payloads_not_rescanned=True,
        complete_final_source_closure=False, actual_final_candidate_admitted=False,
        model_loaded=False, new_games=0, optimizer_updates=0,
        allocation=allocation, seconds=time.monotonic()-started)
    write(out / 'summary-private.json', result)
    for path in (*out.glob('*-private.json'),
            ROOT / 'tools/curriculum_confirmation_json_sources_v1.py',
            ROOT / 'tools/curriculum_confirmation_contract_v1.py',
            ROOT / 'tools/inventory_curriculum_source_delta_v1.py'):
        files[str(path.resolve())] = sha(path)
    write(out / 'completion-binding-private.json', dict(files=files, actual_final_candidate_admitted=False))
    print(json.dumps({k:v for k,v in result.items() if k != 'allocation'}), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--inventory', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(); observe(args.inventory, args.out)
