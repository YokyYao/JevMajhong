"""Assemble immutable complete label sets with disjoint original source groups."""
import json
from pathlib import Path

from .collect import digest, write
from .prepare import checksum


def label_set_contract(sets):
    if not {'legacy4', 'target16'} <= set(sets):
        raise ValueError('The registered original label sets must be retained')
    result = {}
    for tag, spec in sets.items():
        if tag in ('legacy4', 'target16'):
            count, worlds = (264, 4) if tag == 'legacy4' else (44, 16)
            if spec.get('positions', count) != count or spec['future_worlds'] != worlds:
                raise ValueError('Original label-set contract changed')
        else:
            count, worlds = spec.get('positions'), spec.get('future_worlds')
            if type(count) is not int or not 1 <= count <= 2000 or type(worlds) is not int or not 1 <= worlds <= 32:
                raise ValueError('Additional label sets require explicit count and MC-world contracts')
        result[tag] = (count, worlds)
    return result


def assemble(config):
    out = Path(config['out'])
    sets = config['label_sets']
    from .context import PUBLIC_SETTLEMENT_VERSIONS
    contracts = label_set_contract(sets)
    if config.get('input_version') not in PUBLIC_SETTLEMENT_VERSIONS:
        raise ValueError('Mixed pilot requires verified public-settlement inputs')
    selected, public_rows, seen, groups, counts = [], {}, set(), {'train': set(), 'validation': set()}, {}
    sources, label_hashes = {}, {}
    seed_groups = {'train': set(), 'validation': set()}
    for tag, spec in sets.items():
        root = Path(spec['root'])
        positions = json.loads((root / 'selected-positions-private.json').read_text())
        context = json.loads((root / 'public-context-private.json').read_text())
        if (context['schema'] != 'current-public-win-context-v1'
                or not context['source_prefix_only'] or context['future_results_used']):
            raise ValueError('Public context provenance mismatch')
        expected_count, worlds = contracts[tag]
        if len(positions) != expected_count:
            raise ValueError('Original source count or MC count differs')
        verification = None
        if tag != 'legacy4':
            verification = json.loads((root / 'label-protocol-verification-private.json').read_text())
            if verification['status'] != 'frozen_label_protocol_verified':
                raise ValueError('Label generation protocol was not verified')
            if tag != 'target16' and (verification.get('positions_sha256') != digest(positions)
                    or verification.get('positions') != expected_count or verification.get('future_worlds') != worlds):
                raise ValueError('Additional label-set verification differs from its contract')
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
            if tag not in ('legacy4', 'target16') and verification.get('labels_sha256', {}).get(rid) != label_hashes[rid]:
                raise ValueError('An additional original label changed after protocol verification')
            if row['split'] not in seed_groups:
                raise ValueError('Unexpected original source split')
            seed_groups[row['split']].add(row['seed'])
            groups[row['split']].add(row['group_id'])
            selected.append(row)
    if groups['train'] & groups['validation'] or seed_groups['train'] & seed_groups['validation']:
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
