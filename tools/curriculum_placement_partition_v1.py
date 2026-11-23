"""Current partition membership independent of inherited source split tags.

Callers must bind both original data files and this manifest before fitting.
Source metadata is preserved, not silently rewritten in the raw corpus.
"""
from collections import Counter
import copy
import hashlib
import json

from tools import curriculum_placement_exposures_v1 as exposure
from tools import curriculum_placement_metrics_v1 as metrics

METHOD = 'placement_exact_current_file_membership_preserving_source_metadata_v1'
DERIVED = 'placement_current_partition_view_v1'


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    separators=(',', ':')).encode()).hexdigest()


def membership(pools):
    if set(pools) != {'train', 'validation'}:
        raise ValueError('Both complete original current partitions required')
    result = dict(method=METHOD, partitions={})
    groups = {}; all_ids = set()
    for split, rows in pools.items():
        if not rows:
            raise ValueError('Empty partition is not admitted')
        fingerprints = {}; order = []; groups[split] = set(); legacy = Counter()
        for row in rows:
            meta = row['_meta']; rid = meta['id']; group = meta['group_id']
            if (not isinstance(rid, str) or not rid or rid in all_ids
                    or not isinstance(group, str) or not group
                    or DERIVED in meta or exposure.REPEAT_FIELD in meta):
                raise ValueError('Unique pristine IDs and families across current files required')
            all_ids.add(rid); groups[split].add(group); order.append(rid)
            fingerprints[rid] = digest(row)
            legacy[str(meta['split']) if 'split' in meta else '<absent>'] += 1
        result['partitions'][split] = dict(records=len(rows), row_sha256=fingerprints,
            order_sha256=digest(order), groups=sorted(groups[split]), legacy_split_counts=dict(legacy))
    if groups['train'] & groups['validation']:
        raise ValueError('A current source family crossed train and validation')
    result['id_overlap'] = 0; result['group_overlap'] = 0
    return result


def require_partition(rows, manifest, split):
    if manifest['method'] != METHOD or split not in ('train', 'validation'):
        raise ValueError('Bound exact current-membership manifest required')
    spec = manifest['partitions'][split]
    if (manifest['id_overlap'] != 0 or manifest['group_overlap'] != 0
            or set(manifest['partitions']['train']['groups']) & set(manifest['partitions']['validation']['groups'])
            or set(manifest['partitions']['train']['row_sha256']) & set(manifest['partitions']['validation']['row_sha256'])
            or len(rows) != spec['records'] or digest([r['_meta']['id'] for r in rows]) != spec['order_sha256']):
        raise ValueError('Exact admitted current partition and ordering required')
    for row in rows:
        meta = row['_meta']
        if (DERIVED in meta or exposure.REPEAT_FIELD in meta
                or digest(row) != spec['row_sha256'].get(meta['id'])):
            raise ValueError('Original record identity/content changed or a view/copy was supplied')


def partition_views(rows, manifest, split):
    require_partition(rows, manifest, split)
    result = []
    for row in rows:
        meta = row['_meta']
        derived = dict(original_split_present='split' in meta,
            original_split_value=copy.deepcopy(meta.get('split')), current_partition=split,
            membership_method=METHOD)
        result.append(dict(row, _meta=dict(meta, split=split, **{DERIVED: derived})))
    return result


def expand_train(rows, plan, manifest):
    # Compatibility views satisfy the earlier helper's split check. Restore
    # original metadata on every output; no view becomes a raw training row.
    views = partition_views(rows, manifest, 'train')
    expanded = exposure.expand_train(views, plan)
    originals = {r['_meta']['id']: r for r in rows}
    result = list(rows)
    for replica in expanded[len(rows):]:
        meta = replica['_meta']; provenance = meta[exposure.REPEAT_FIELD]
        original = originals[provenance['original_id']]
        replica['_meta'] = dict(copy.deepcopy(original['_meta']), id=meta['id'],
                               **{exposure.REPEAT_FIELD: provenance})
        result.append(replica)
    return result


def profile(records, predictions, base, old_ce, manifest):
    views = partition_views(records, manifest, 'validation')
    result = metrics.profile(views, predictions, base, old_ce)
    result.update(validation_current_membership_verified=True,
        inherited_source_split_tag_not_current_partition=True,
        inherited_validation_source_split_counts=manifest['partitions']['validation']['legacy_split_counts'],
        current_membership_method=METHOD)
    return result

