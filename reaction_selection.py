"""Register reaction-only checkpoint selection without changing frozen trainers."""
from collections import Counter


def validate_recipe(config, recipe):
    expected = {
        'eval_data': config['data'] + '/validation.jsonl',
        'eval_transfer_suite': config['data'] + '/reaction-validation',
        'early_stop_metric': 'transfer_nll',
        'eval_records': 0,
    }
    if config.get('selection_scope') != 'reaction':
        raise ValueError('Explicit reaction selection scope required')
    if any(recipe.get(k) != v for k, v in expected.items()):
        raise ValueError('Actual trainer must select full reaction transfer NLL')


def reaction_subset(records):
    ids = [r['_meta']['id'] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate validation IDs')
    selected = []
    actions = Counter()
    for record in records:
        meta = record['_meta']
        if meta['split'] != 'validation' or meta['view'] != 'native':
            raise ValueError('Only already-opened native validation is eligible')
        if meta['phase'] != 'reaction':
            continue
        q = record['questions']['action']
        kind = q['criteria'][q['label']]['type']
        if kind not in ('chi', 'pon', 'none'):
            raise ValueError('Foreign reaction label')
        selected.append(record)
        actions[kind] += 1
    if set(actions) != {'chi', 'pon', 'none'}:
        raise ValueError('Reaction validation must cover chi, pon and pass')
    return selected, dict(actions)


def verify_benchmark_contract(records):
    """Exercise real benchmark row construction without a model or a GPU."""
    from jevany.benchmark import prediction_rows, summarize
    rows = []
    for record in records:
        probabilities = {}
        for qid, q in record['questions'].items():
            keys = list(q['criteria'])
            probabilities[qid] = {key: 1. / len(keys) for key in keys}
        rows.extend(prediction_rows(record, {'probabilities': probabilities}))
    report = summarize(rows)
    if report['clean']['n'] != len(records):
        raise ValueError('Selection suite benchmark coverage differs')
    return report['clean']['n']
