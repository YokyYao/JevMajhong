"""Read-only reusable sidecar index of the existing audited curriculum union.

This is catalog construction, not another native/token/model audit. Structural
hand hints do not certify yaku, route value, labels or student competence.
"""
import argparse
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha

HONORS = ('E', 'S', 'W', 'N', 'P', 'F', 'C')
TERMINALS = (0, 8, 9, 17, 18, 26, 27, 28, 29, 30, 31, 32, 33)


def tile_type(tile):
    if tile in HONORS:
        return 27 + HONORS.index(tile)
    token = tile.removesuffix('r')
    rank, suit = int(token[0]), token[1]
    rank = rank or 5
    if suit not in 'mpsz' or not 1 <= rank <= (7 if suit == 'z' else 9):
        raise ValueError('Invalid public tile spelling')
    return {'m': 0, 'p': 9, 's': 18, 'z': 27}[suit] + rank - 1


@lru_cache(maxsize=160000)
def closed_structure(counts):
    from mahjong.shanten import Shanten
    # These distances describe the current concealed shape, not eventual yaku.
    normal = Shanten.calculate_shanten(list(counts), use_chiitoitsu=False, use_kokushi=False)
    seven = Shanten.calculate_shanten_for_chiitoitsu_hand(list(counts))
    kokushi = Shanten.calculate_shanten_for_kokushi_hand(list(counts))
    baseline = (3, 1, 1, 1, 1, 1, 1, 1, 3)
    chuuren = {}
    for offset, suit in ((0, 'm'), (9, 'p'), (18, 's')):
        chuuren[suit] = sum(max(0, baseline[i] - counts[offset+i]) for i in range(9))
    return dict(regular_shanten=normal, chiitoitsu_shanten=seven, kokushi_shanten=kokushi,
                chuuren_base_missing_tiles=chuuren)


def index_record(row, split, source, line):
    state, meta = row['state'], row['_meta']
    question = row['questions']['action']
    menu = question['criteria']
    counts = [0]*34
    for tile in state['hand'].split():
        counts[tile_type(tile)] += 1
    if max(counts) > 4:
        raise ValueError('More than four copies in a public concealed hand')
    seat = state['seat']
    own_melds = state['melds'][seat]
    offered = sorted({item['type'] for item in menu.values()})
    hard = question.get('target') is None
    # A soft-target argmax key is not relabeled as an actual teacher selection.
    selected = menu[question['label']]['type'] if hard else None
    hints = dict(concealed_tiles=sum(counts), own_meld_count=len(own_melds),
                 own_meld_types=[m['type'] for m in own_melds],
                 pair_or_triplet_types=sum(n >= 2 for n in counts),
                 triplet_types=sum(n >= 3 for n in counts),
                 distinct_terminal_honor_types=sum(counts[t] > 0 for t in TERMINALS),
                 dragon_pair_or_triplet_types=sum(counts[t] >= 2 for t in (31, 32, 33)),
                 dragon_triplet_types=sum(counts[t] >= 3 for t in (31, 32, 33)))
    tags = ['basic_replay' if question['src'].startswith('curriculum2:') else
            'conditional_simulation' if question['src'].startswith('curriculum3:mc:') else
            'risk' if 'under_riichi' in question['src'] else 'ordinary_completion']
    tags.extend('offered:'+kind for kind in offered)
    if own_melds:
        hints['closed_shape_distances_available'] = False
    elif sum(counts) in (13, 14):
        hints.update(closed_structure(tuple(counts)), closed_shape_distances_available=True)
        for name in ('kokushi', 'chiitoitsu'):
            if hints[name+'_shanten'] <= 1:
                tags.append('near_shape:'+name)
        for suit, missing in hints['chuuren_base_missing_tiles'].items():
            if missing <= 2:
                tags.append('near_base_pattern:chuuren_'+suit)
        if hints['triplet_types'] >= 3:
            tags.append('triplet_rich_concealed_not_certified_suuankou')
        if hints['dragon_pair_or_triplet_types'] == 3:
            tags.append('three_dragons_supported_not_certified_daisangen')
    else:
        hints['closed_shape_distances_available'] = False
        hints['unavailable_reason'] = 'concealed_count_not13_or14'
    return dict(record_id=meta['id'], group_id=meta['group_id'], partition=split,
                source_file=source, source_line_1based=line, source=question['src'],
                inherited_producer_split=meta.get('split'), phase=meta.get('phase'),
                label_method='observed_hard' if hard else 'conditional_soft_estimate',
                label_source_type=meta.get('label_type'), offered_operations=offered,
                observed_hard_selected_operation=selected, tags=tags,
                public_structure=hints, quality='inherited_audited_source_not_new_yaku_certification')


def catalog(config):
    gpu_only()
    c = read(config)
    if c['method'] != 'audited_curriculum_union_reusable_sidecar_catalog_v1':
        raise ValueError('Unknown catalog recipe')
    source_audit = ROOT/c['source_audit']
    audit = read(source_audit)
    if audit['status'] != 'exact_progression_union_original_rows_encoder_masks_and_split_audit_passed_not_fit_admitted':
        raise ValueError('Use the already audited original union')
    out = ROOT/c['out']
    out.mkdir(parents=True, exist_ok=False)
    files = {str(p.resolve()): sha(p) for p in (Path(__file__), config, source_audit,
             ROOT/'cluster/curriculum-data-catalog-v1.slurm',
             ROOT/'docs/experiments/plans/curriculum-data-catalog-20261123.md')}
    write(out/'registered-private.json', dict(method=c['method'], config=c, files=files,
          job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname(),
          source_files=audit['files'], changes_to_active_fit=False))
    totals = {}; group_sets = {}; outputs = {}
    try:
        for split in ('train', 'validation'):
            spec = audit['files'][split]
            path = ROOT/spec['path']
            # Hash once while constructing the index; original files are read-only.
            digest = hashlib.sha256()
            ids = set(); groups = set(); source_counts = Counter(); tag_counts = Counter()
            offered = Counter(); selected = Counter(); modes = Counter(); phase = Counter()
            tagged_groups = defaultdict(set); inherited = Counter(); n = 0
            target = out/f'{split}-index-private.jsonl'
            with path.open('rb') as source, target.open('x') as stream:
                for n, raw in enumerate(source, 1):
                    digest.update(raw)
                    item = index_record(json.loads(raw), split, spec['path'], n)
                    if item['record_id'] in ids:
                        raise ValueError('Duplicate record identity in catalog source')
                    ids.add(item['record_id']); groups.add(item['group_id'])
                    source_counts[item['source']] += 1
                    tag_counts.update(item['tags']); offered.update(item['offered_operations'])
                    modes[item['label_method']] += 1; phase[str(item['phase'])] += 1
                    inherited[str(item['inherited_producer_split'])] += 1
                    if item['observed_hard_selected_operation'] is not None:
                        selected[item['observed_hard_selected_operation']] += 1
                    for tag in item['tags']:
                        tagged_groups[tag].add(item['group_id'])
                    stream.write(json.dumps(item, separators=(',', ':'))+'\n')
                    if n % 20000 == 0:
                        print(json.dumps(dict(partition=split, indexed_records=n)), flush=True)
            if digest.hexdigest() != spec['sha256'] or n != spec['records'] or len(groups) != spec['groups']:
                raise ValueError('Source identity/count changed during sidecar construction')
            group_sets[split] = groups
            totals[split] = dict(records=n, unique_groups=len(groups), sources=dict(source_counts),
                tags=dict(tag_counts), unique_groups_by_tag={k: len(v) for k, v in tagged_groups.items()},
                offered_operation_rows=dict(offered), observed_hard_selected_operations=dict(selected),
                label_methods=dict(modes), phases=dict(phase), inherited_producer_splits=dict(inherited))
            outputs[split] = dict(path=str(target.relative_to(ROOT)), sha256=sha(target), records=n)
        if group_sets['train'] & group_sets['validation']:
            raise ValueError('Catalog cannot cross existing family partitions')
        if any(sha(p) != h for p, h in files.items()):
            raise ValueError('Catalog source code/config changed during execution')
        summary = dict(status='actual_existing_union_sidecar_catalog_complete_not_new_training_data',
            method=c['method'], coverage=totals, source_files={s: audit['files'][s] for s in ('train', 'validation')},
            indexes=outputs, original_quarantine_excluded=audit['original_quarantine_preserved'],
            partition_authority='current_bound_source_file_not_inherited_origin_marker',
            source_inputs_labels_and_partitions_unchanged=True, cross_split_group_overlap=0,
            natural_game_rates_not_estimated=True, shape_tags_not_yaku_or_route_optimality_labels=True,
            new_simulated_games=0, new_teacher_labels=0, student_updates=0,
            active_fit_not_modified=True, raw_indexes_private=True,
            job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
        write(out/'summary-private.json', summary)
        write(out/'completion-binding-private.json', dict(files=files))
        print(json.dumps(dict(status=summary['status'], records={s: totals[s]['records'] for s in totals})), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(type=type(error).__name__, error=str(error)))
        raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    args = p.parse_args()
    catalog(args.config.resolve())
