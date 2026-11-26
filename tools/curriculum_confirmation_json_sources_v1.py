"""File-backed JSON seed evidence, not complete-ancestry or final admission.

Only exact, hash-bound protected-range declarations can be classified apart
from data occurrences. The caller must still establish complete source-graph
coverage and independently verify the actual source-family exclusions.
"""
import hashlib
import json
import math
from pathlib import Path
import re


def _object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError('Duplicate JSON keys cannot support source evidence')
        result[key] = value
    return result


def _constant(value):
    raise ValueError('Nonfinite JSON constants cannot support source evidence')


def _pointer_token(value):
    return value.replace('~', '~0').replace('/', '~1')


def _findings(value, pattern, pointer=''):
    if isinstance(value, dict):
        for key, child in value.items():
            where = pointer + '/' + _pointer_token(key)
            for match in pattern.finditer(key):
                yield dict(pointer=where, kind='key', seed=int(match[0]))
            yield from _findings(child, pattern, where)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from _findings(child, pattern, pointer + '/' + str(index))
    elif type(value) in (int, float, str):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError('Nonfinite JSON numbers cannot support source evidence')
        for match in pattern.finditer(str(value)):
            yield dict(pointer=pointer, kind=type(value).__name__, seed=int(match[0]))


def _protected_range(value, pointer, seed_start, groups):
    if not re.fullmatch(r'/(?:config/)?protected_source_ranges/(?:0|[1-9][0-9]*)/start', pointer):
        raise ValueError('Only explicit protected-source-range start pointers may be declared')
    current = value
    for token in pointer.split('/')[1:-1]:
        if isinstance(current, list):
            current = current[int(token)]
        else:
            current = current[token]
    if (not isinstance(current, dict) or set(current) != {'start', 'count'}
            or type(current['start']) is not int or type(current['count']) is not int
            or current != dict(start=seed_start, count=groups)):
        raise ValueError('Exact protected reserved range required at a declared pointer')


def scan_json_artifacts(paths, protected_declarations=None, seed_start=2026300000, groups=32):
    if type(seed_start) is not int or type(groups) is not int or groups <= 0:
        raise ValueError('Explicit positive reserved logical-seed interval required')
    canonical = sorted({Path(path).resolve() for path in paths})
    if not canonical or any(not path.is_file() or path.suffix != '.json' for path in canonical):
        raise ValueError('Nonempty explicit JSON file inventory required')
    declarations = {}
    for name, declaration in (protected_declarations or {}).items():
        name = str(Path(name).resolve())
        if name in declarations:
            raise ValueError('Duplicate canonical declaration path')
        if (set(declaration) != {'sha256', 'start_pointers'}
                or not re.fullmatch('[0-9a-f]{64}', declaration['sha256'])
                or not isinstance(declaration['start_pointers'], list)
                or not declaration['start_pointers']
                or any(not isinstance(p, str) for p in declaration['start_pointers'])
                or len(set(declaration['start_pointers'])) != len(declaration['start_pointers'])):
            raise ValueError('Each declaration requires an exact hash and unique nonempty pointers')
        declarations[name] = declaration
    if not set(declarations) <= {str(path) for path in canonical}:
        raise ValueError('Every declared artifact must be in the actual inventory')
    pattern = re.compile(r'(?<![0-9])(?:' + '|'.join(str(seed) for seed in range(seed_start, seed_start + groups)) + r')(?![0-9])')
    receipts = []
    for path in canonical:
        before = path.stat()
        raw = path.read_bytes()
        checksum = hashlib.sha256(raw).hexdigest()
        value = json.loads(raw, object_pairs_hook=_object, parse_constant=_constant)
        after = path.stat()
        findings = list(_findings(value, pattern))
        declaration = declarations.get(str(path))
        allowed = set()
        if declaration:
            if declaration['sha256'] != checksum:
                raise ValueError('A reviewed declaration artifact changed')
            for pointer in declaration['start_pointers']:
                _protected_range(value, pointer, seed_start, groups)
                expected = dict(pointer=pointer, kind='int', seed=seed_start)
                if findings.count(expected) != 1:
                    raise ValueError('Each reviewed declaration must match one actual integer occurrence')
                allowed.add(pointer)
        reviewed = [row for row in findings if row['pointer'] in allowed
                    and row['kind'] == 'int' and row['seed'] == seed_start]
        unreviewed = [row for row in findings if row not in reviewed]
        receipts.append(dict(path=str(path), sha256=checksum, bytes_scanned=len(raw),
            stable_during_read=(before.st_size == after.st_size == len(raw)
                and before.st_mtime_ns == after.st_mtime_ns),
            classification=('unreviewed_reserved_occurrences' if unreviewed else
                'only_exact_reviewed_protected_range_declarations' if reviewed else 'no_reserved_occurrences'),
            reserved_occurrences=len(findings), reviewed_declarations=reviewed,
            unreviewed_occurrences=len(unreviewed), first_unreviewed_private=unreviewed[:32]))
    return dict(method='explicit_JSON_semantic_reserved_seed_occurrences_v1',
        seed_start=seed_start, seed_groups=groups, files=receipts, total_files=len(receipts),
        total_bytes=sum(row['bytes_scanned'] for row in receipts),
        reserved_occurrences=sum(row['reserved_occurrences'] for row in receipts),
        reviewed_declaration_occurrences=sum(len(row['reviewed_declarations']) for row in receipts),
        unreviewed_reserved_occurrences=sum(row['unreviewed_occurrences'] for row in receipts),
        all_inventoried_files_stable=all(row['stable_during_read'] for row in receipts),
        checked_files_only_not_complete_source_graph=True,
        source_family_exclusion_not_proved_by_this_function=True,
        reserved_native_resets_or_games=0, confirmation_admitted=False)
