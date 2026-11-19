"""Assemble immutable complete label sets with disjoint original source groups."""
import json
from pathlib import Path

from .collect import digest, write
from .prepare import checksum


def assemble(config):
    out = Path(config['out'])
    sets = config['label_sets']
    if set(sets) != {'legacy4', 'target16'} or config.get('input_version') != 'endgame-public-v3':
        raise ValueError('Mixed pilot requires the registered original4/new16 label sets')
    target = Path(sets['target16']['root'])
    verification = json.loads((target / 'label-protocol-verification-private.json').read_text())
    if verification['status'] != 'frozen_label_protocol_verified':
        raise ValueError('Target label generation protocol was not verified')
    selected, public_rows, seen, groups, counts = [], {}, set(), {'train': set(), 'validation': set()}, {}
    sources, label_hashes = {}, {}
    for tag, spec in sets.items():
        root = Path(spec['root'])
        positions = json.loads((root / 'selected-positions-private.json').read_text())
        context = json.loads((root / 'public-context-private.json').read_text())
        if (context['schema'] != 'current-public-win-context-v1'
                or not context['source_prefix_only'] or context['future_results_used']):
            raise ValueError('Public context provenance mismatch')
        expected_count = 264 if tag == 'legacy4' else 44
        if len(positions) != expected_count or spec['future_worlds'] != (4 if tag == 'legacy4' else 16):
            raise ValueError('Original source count or MC count differs')
        counts[tag] = len(positions)
        for original in positions:
            row = dict(original, _label_set=tag)
            rid = digest([row['seed'], row['step'], row['seat'], row['request_sha256']])
            if rid in seen:
                raise ValueError('Repeated terminal position across label sets')
            seen.add(rid)
            if context['rows'][rid]['source_request_sha256'] != row['request_sha256']:
                raise ValueError('Public feature/source identity mismatch')
            public_rows[rid] = context['rows'][rid]
            source = root / 'sources' / f"{row['seed']}.json"
            if row['seed'] in sources and sources[row['seed']].resolve() != source.resolve():
                raise ValueError('Conflicting native source identity')
            sources[row['seed']] = source
            label = root / 'labels' / (rid + '.json')
            data = json.loads(label.read_text())
            if (data['status'] != 'complete_terminal_menu' or data['worlds_per_action'] != spec['future_worlds']
                    or data['source_request_sha256'] != row['request_sha256']):
                raise ValueError('Incomplete or mismatched original label')
            label_hashes[rid] = checksum(label)
            groups[row['split']].add(row['group_id'])
            selected.append(row)
    if groups['train'] & groups['validation']:
        raise ValueError('Mixed terminal sources cross train/validation')
    manifest = out / 'selected-positions-private.json'
    if manifest.exists() and json.loads(manifest.read_text()) != selected:
        raise ValueError('Existing assembled manifest differs')
    for directory in ('labels', 'sources'):
        (out / directory).mkdir(parents=True, exist_ok=True)
    for row in selected:
        rid = digest([row['seed'], row['step'], row['seat'], row['request_sha256']])
        original = Path(sets[row['_label_set']]['root']) / 'labels' / (rid + '.json')
        link = out / 'labels' / (rid + '.json')
        if link.exists() and link.resolve() != original.resolve():
            raise ValueError('Existing label alias differs')
        if not link.exists():
            link.symlink_to(original.resolve())
    for seed, original in sources.items():
        link = out / 'sources' / f'{seed}.json'
        if link.exists() and link.resolve() != original.resolve():
            raise ValueError('Existing source alias differs')
        if not link.exists():
            link.symlink_to(original.resolve())
    write(manifest, selected)
    write(out / 'public-context-private.json', {'schema': 'current-public-win-context-v1',
          'rows': public_rows, 'source_prefix_only': True, 'future_results_used': False})
    result = {'status': 'immutable_label_sets_assembled', 'positions': len(selected),
        'label_set_positions': counts, 'positions_sha256': digest(selected),
        'groups': {s: len(g) for s, g in groups.items()}, 'group_overlap': 0,
        'original_label_sha256': label_hashes,
        'all_original_outcomes_preserved': True, 'optimizer_updates': 0}
    write(out / 'assembly-private.json', result)
    return result
