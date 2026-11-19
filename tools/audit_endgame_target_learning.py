"""Fixed-data target learning and a single original-instruction ablation."""
import argparse
from collections import Counter, defaultdict
import copy
import gc
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def validate_config(config):
    expected = {'schema': 'endgame-target-learning-header-diagnostic-v1',
        'research_step': 72, 'training_ordinary_fourth_positions': 55,
        'validation_positions': 186, 'header_variant': 'released_teacher_instructions_only',
        'baseline_repeat_controls': 8, 'control_selection': 'public_fixed_hash',
        'control_max_abs_difference': 1e-7, 'max_model_queries': 304,
        'new_match_simulations': 0, 'diagnostic_only': True}
    if any(config.get(k) != v for k, v in expected.items()) or any(config.get(k) is not False
            for k in ('fit_registered', 'promotion_registered', 'independent_confirmation',
                      'export_registered', 'raw_publication')):
        raise ValueError('Only the registered non-promoting304-query diagnostic is enabled')


def original_instruction_variant(payload, teacher):
    """Change exactly one instruction field; retain current state and legal features."""
    value = teacher['questions']['action'].get('instructions')
    if not isinstance(value, str) or not value:
        raise ValueError('Original teacher instruction is missing')
    if set(payload['questions']) != {'action'}:
        raise ValueError('Only the current single action question is registered')
    result = copy.deepcopy(payload)
    result['questions']['action']['instructions'] = value
    return result


def validation_metrics(records, predictions, reference):
    from jevmahjong.endgame.fit import target_ce, original_replay_ce, reference_kl_group
    from jevmahjong.endgame.native_terminal import is_native_terminal
    from jevmahjong.endgame.public_policy import action_keys
    from jevmahjong.endgame.reference import distribution, restrict_distribution
    if set(predictions) != {r['_meta']['id'] for r in records}:
        raise ValueError('Validation prediction identity set differs')
    ce, kl, utility = defaultdict(list), defaultdict(list), defaultdict(list)
    escape = defaultdict(list); original_escape = defaultdict(list)
    third, original_third = [], []
    for row in records:
        meta, q = row['_meta'], row['questions']['action']; rid = meta['id']
        p = predictions[rid]; distribution(p, list(q['criteria']))
        eligible = action_keys(row); chosen = max(eligible, key=p.get)
        prior = reference[rid]['probabilities']; baseline = max(eligible, key=prior.get)
        ce[q['src']].append(target_ce(q, p))
        if q['src'].startswith('replay_') or is_native_terminal(meta):
            current = restrict_distribution(p, eligible)
            group = reference_kl_group(q['src'], meta)
            kl[group].append(sum(v * math.log(max(v, 1e-30) / max(current[k], 1e-30))
                                 for k, v in prior.items() if v > 0))
        if q['src'] == 'endgame_terminal':
            values = meta['expected_terminal_utilities']; utility[meta['bucket']].append(values[chosen])
            if meta['source_initial_placement'] == 4:
                cohort = 'native_fourth' if is_native_terminal(meta) else 'existing'
                escape[cohort].append(meta['terminal_escape4_probabilities'][chosen])
                original_escape[cohort].append(meta['terminal_escape4_probabilities'][baseline])
            if meta['source_initial_placement'] == 3:
                third.append(values[chosen]); original_third.append(values[baseline])
    if (len(records) != 186 or {k: len(v) for k, v in escape.items()} != {'existing': 43, 'native_fourth': 8}
            or {k: len(v) for k, v in kl.items()} != {'original': 64, 'trailing': 16, 'leading': 16}
            or set(utility) != {'win', 'third', 'fourth', 'hold'} or len(third) != 23 or len(utility['hold']) != 12):
        raise ValueError('Legacy43/new8/third23/hold12 or64/16/16 validation coverage changed')
    mean = lambda values: sum(values) / len(values)
    strata = {k: mean(v) for k, v in utility.items()}
    return {'records': 186, 'original_expert_ce': original_replay_ce(ce),
        'kl': {k: {'positions': len(v), 'mean': mean(v)} for k, v in kl.items()},
        'fourth_escape': {k: {'positions': len(v), 'candidate': mean(v),
                             'original_policy': mean(original_escape[k])} for k, v in escape.items()},
        'third_pursuit': {'positions': 23, 'candidate': mean(third), 'original_policy': mean(original_third)},
        'utility_by_stratum': strata, 'equally_stratified_utility': mean(list(strata.values()))}


def verify_logged_metrics(current, logged):
    comparisons = [(current['original_expert_ce'], logged['replay_target_ce']),
        (current['fourth_escape']['existing']['candidate'], logged['fourth_escape_validation']['candidate']),
        (current['fourth_escape']['native_fourth']['candidate'], logged['native_fourth_escape_validation']['candidate']),
        (current['third_pursuit']['candidate'], logged['third_pursuit_validation']['candidate'])]
    comparisons += [(current['utility_by_stratum'][k], v) for k, v in logged['utility_by_stratum'].items()]
    comparisons += [(current['kl'][k]['mean'], logged[field]) for k, field in (
        ('original', 'replay_reference_kl'), ('trailing', 'native_reference_kl'), ('leading', 'leading_native_reference_kl'))]
    if any(abs(a - b) > 1e-12 for a, b in comparisons):
        raise ValueError('Cached predictions do not reproduce the recorded training metrics')


def training_metrics(records, predictions, reference):
    from jevmahjong.endgame.fit import target_ce
    from jevmahjong.endgame.native_terminal import is_native_terminal
    from jevmahjong.endgame.public_policy import action_keys
    from jevmahjong.endgame.reference import distribution, escape_supported_keys
    if set(predictions) != {r['_meta']['id'] for r in records}:
        raise ValueError('Training diagnostic prediction set differs')
    totals = Counter(); sums = Counter(); cohorts = defaultdict(Counter)
    for row in records:
        meta, q = row['_meta'], row['questions']['action']; rid = meta['id']
        p = predictions[rid]; distribution(p, list(q['criteria']))
        prior = reference[rid]['probabilities']; eligible = action_keys(row)
        chosen = max(eligible, key=p.get); baseline = max(eligible, key=prior.get)
        values, escape = meta['expected_terminal_utilities'], meta['terminal_escape4_probabilities']
        safe = escape_supported_keys(prior, escape, eligible)
        best = max(values[k] for k in safe)
        strict = best > values[baseline] + 1e-12
        hit = strict and values[chosen] >= best - 1e-12 and escape[chosen] >= escape[baseline] - 1e-12
        group = 'native' if is_native_terminal(meta) else 'existing'
        totals['positions'] += 1; totals['target_top1_matches'] += chosen == q['label']
        totals['strict_improvement_positions'] += strict; totals['strict_target_hits'] += hit
        totals['escape_violations_vs_original_greedy'] += escape[chosen] < escape[baseline] - 1e-12
        cohorts[group]['positions'] += 1; cohorts[group]['strict_positions'] += strict; cohorts[group]['strict_hits'] += hit
        sums['target_ce'] += target_ce(q, p)
        sums['terminal_utility'] += values[chosen]; sums['escape'] += escape[chosen]
        if strict:
            sums['strict_target_probability'] += sum(p[k] for k, v in q['target'].items() if v > 0)
    n = len(records); strict_n = totals['strict_improvement_positions']
    return {'counts': dict(totals), 'by_source': {k: dict(v) for k, v in cohorts.items()},
        'mean_target_ce': sums['target_ce'] / n, 'mean_finite_utility': sums['terminal_utility'] / n,
        'mean_finite_escape': sums['escape'] / n,
        'mean_strict_target_probability': sums['strict_target_probability'] / strict_n if strict_n else None}


def reconstruct_teachers(records, run, run_config, reference):
    from desktop.jev_sidecar.unified_policy import serving_request
    from jevmahjong.endgame.prepare import read_jsonl
    from jevmahjong.endgame.native_retention import load_native_pairs
    from jevmahjong.endgame.public_policy import original_policy_keys
    from jevmahjong.endgame.reference import inference_request, reused_reference_prior
    needed = {r['_meta']['id'] for r in records if not r['_meta']['id'].startswith('endgame:')}
    originals, source_paths = {}, set()
    for field in ('replay_train', 'replay_validation'):
        path = Path(run_config[field]); source_paths.add(path)
        for raw in read_jsonl(path):
            rid = raw['_meta']['id']
            if rid in needed:
                payload = inference_request(raw)
                if rid in originals and originals[rid] != payload:
                    raise ValueError('Ambiguous original replay input')
                originals[rid] = payload
    for row in load_native_pairs(run_config):
        if row['id'] in needed: originals[row['id']] = row['raw_request']
    for field in ('native_context_root', 'native_leading_context_root'):
        source_paths.update(Path(run_config[field]) / name for name in ('paired-contexts-private.json', 'summary-private.json'))
    if set(originals) != needed:
        raise ValueError('An original teacher input is missing')
    teachers = {}
    for row in records:
        rid = row['_meta']['id']
        if rid.startswith('endgame:'):
            path = run / 'labels' / (rid.split(':', 1)[1] + '.json'); source_paths.add(path)
            raw = copy.deepcopy(json.loads(path.read_text())['source']['request'])
            raw['state'].pop('endgame_context', None)
        else:
            raw = originals[rid]
        teacher = serving_request(raw)
        keys = list(row['questions']['action']['criteria'])
        reused_reference_prior(reference[rid], inference_request(row), teacher, original_policy_keys(teacher), keys)
        teachers[rid] = teacher
    return teachers, source_paths


def drift_summary(records, predictions, reference):
    from jevmahjong.endgame.native_terminal import is_native_terminal
    from jevmahjong.endgame.public_policy import action_keys
    result = defaultdict(Counter)
    for row in records:
        meta, q = row['_meta'], row['questions']['action']
        if q['src'] != 'endgame_terminal' or meta['source_initial_placement'] != 4: continue
        rid = meta['id']; eligible = action_keys(row); prior = reference[rid]['probabilities']
        a, b, c = (max(eligible, key=p.get) for p in (prior, predictions['initial'][rid], predictions['trained'][rid]))
        cohort = 'native_fourth' if is_native_terminal(meta) else 'existing'
        escape = meta['terminal_escape4_probabilities']; counter = result[cohort]
        counter['positions'] += 1
        counter['initial_lower_escape_than_original'] += escape[b] < escape[a] - 1e-12
        counter['trained_lower_escape_than_original'] += escape[c] < escape[a] - 1e-12
        counter['trained_lower_escape_than_initial'] += escape[c] < escape[b] - 1e-12
        if c != a: counter['original_to_trained:' + q['criteria'][a]['type'] + '->' + q['criteria'][c]['type']] += 1
    return {k: dict(v) for k, v in result.items()}


def audit(config):
    validate_config(config)
    from jevmahjong.endgame.collect import digest, write
    from jevmahjong.endgame.prepare import read_jsonl, checksum
    from jevmahjong.endgame.futures import bundle_identity
    from jevmahjong.endgame.reference import inference_request
    from desktop.jev_sidecar.unified_policy import identity
    from tools.audit_endgame_parameter_attribution import saved_predictions, control_comparison
    out, run = Path(config['out']), Path(config['research_run'])
    if (out / 'protocol-private.json').exists() or (out / 'summary-private.json').exists():
        raise ValueError('Inspect existing diagnostic artifacts; never silently restart')
    run_config = json.loads((run / 'config-private.json').read_text())
    baseline = Path(run_config['checkpoint']); candidate = run / 'training-run/checkpoints/step-000072'
    baseline_id, candidate_id = bundle_identity(baseline), bundle_identity(candidate)
    released = identity()
    if baseline_id['identity_sha256'] not in (released['source_identity_sha256'], released['identity_sha256']):
        raise ValueError('Formal baseline differs from released v8')
    if any(baseline_id['files'][k] == candidate_id['files'][k] for k in ('head.pt', 'adapter_model.safetensors')):
        raise ValueError('Research checkpoint must contain actual joint updates')
    fit = json.loads((run / 'fit-result-private.json').read_text())
    selection = json.loads((run / 'training-run/selection.json').read_text())
    postfit = json.loads((run / 'postfit-summary-private.json').read_text())
    if (fit['selection'] != selection or selection['step'] != 0 or fit['eligible_for_development']
            or fit['validation'][-1]['step'] != 72 or not fit['actual_updates_verified']
            or postfit['status'] != 'twelfth_fit_and_selection_audit_complete' or postfit['selected_step'] != 0):
        raise ValueError('Expected the audited, unqualified twelfth72 research checkpoint')
    all_train = read_jsonl(run / 'train.jsonl'); records = read_jsonl(run / 'validation.jsonl')
    train = [r for r in all_train if r['questions']['action']['src'] == 'endgame_terminal_fourth']
    if len(all_train) != 905 or len(train) != 55 or len(records) != 186:
        raise ValueError('Fixed55/186 diagnostic set changed')
    all_records = all_train + records
    reference = json.loads((run / 'reference-targets-private.json').read_text())
    if (reference['status'] != 'complete_frozen_reference' or reference['identity_sha256'] != baseline_id['identity_sha256']
            or reference['input_version'] != 'endgame-trailing-public-v1' or reference['reference_mode'] != 'released_original_policy'
            or set(reference['rows']) != {r['_meta']['id'] for r in all_records}
            or len({r['_meta']['id'] for r in all_records}) != 1091):
        raise ValueError('Original frozen reference identity or complete input set changed')
    for row in all_records:
        if reference['rows'][row['_meta']['id']]['request_sha256'] != digest(inference_request(row)):
            raise ValueError('An original student model input changed')
    cached = {'initial': saved_predictions(run, 0), 'trained': saved_predictions(run, 72)}
    cached_metrics = {k: validation_metrics(records, p, reference['rows']) for k, p in cached.items()}
    for name, logged in (('initial', fit['validation'][0]), ('trained', fit['validation'][-1])):
        verify_logged_metrics(cached_metrics[name], logged)
    teachers, teacher_sources = reconstruct_teachers(records, run, run_config, reference['rows'])
    controls = sorted(records, key=lambda r: digest(['target-learning-header-control-v1', r['_meta']['id']]))[:8]
    upstream = Path(run_config['upstream_source']); sys.path.insert(0, str(upstream))
    paths = set(ROOT.glob('jevmahjong/**/*.py')) | set(ROOT.glob('desktop/jev_sidecar/**/*.py'))
    paths.update((upstream / 'jevany').glob('**/*.py'))
    paths.update(teacher_sources)
    paths.update(ROOT / name for name in ('tools/audit_endgame_target_learning.py',
        'tools/audit_endgame_parameter_attribution.py', 'tools/prepare_native_context_replay.py',
        'configs/training/endgame-target-learning-header-diagnostic-v1.json'))
    paths.update(run / name for name in ('train.jsonl', 'validation.jsonl', 'reference-targets-private.json',
        'config-private.json', 'fit-result-private.json', 'training-run/selection.json', 'postfit-summary-private.json'))
    paths.update(run / 'training-run/training_eval' / f'step-{step:06d}' / 'development/rows.json' for step in (0, 72))
    protocol = {'configuration': config, 'baseline_identity': baseline_id, 'research_identity': candidate_id,
        'source_sha256': {str(p.resolve()): checksum(p) for p in sorted(paths)},
        'control_ids': [r['_meta']['id'] for r in controls],
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'riichienv', 'mahjong')},
        'optimizer_updates': 0, 'new_match_simulations': 0, 'eligible_for_promotion': False}
    write(out / 'protocol-private.json', protocol)
    import torch
    from jevmahjong.policy.routing import load_model
    from jevany.api import validate_distribution
    queries = 0
    def load(path):
        client = load_model(path); client.runtime.model.eval().requires_grad_(False)
        if client.runtime.inference_options.prefix_cache_size != 0:
            raise ValueError('Instruction ablation requires disabled inference prefix cache')
        return client
    def query(client, subset, arm, variant=False):
        nonlocal queries
        result = {}
        for row in subset:
            if queries >= 304: raise ValueError('Registered query budget exceeded')
            rid = row['_meta']['id']; payload = inference_request(row)
            if variant: payload = original_instruction_variant(payload, teachers[rid])
            answer = client.system_one(**payload); queries += 1
            keys = list(row['questions']['action']['criteria'])
            values, _ = validate_distribution(answer['answers']['action']['probabilities'], keys)
            result[rid] = dict(zip(keys, values))
            if len(result) % 25 == 0 or len(result) == len(subset):
                write(out / 'status-private.json', {'status': 'diagnostic_inference', 'arm': arm,
                    'arm_records_complete': len(result), 'arm_records_total': len(subset),
                    'new_model_queries': queries, 'optimizer_updates': 0})
        write(out / (arm + '-predictions-private.json'), result)
        return result
    client = load(baseline)
    repeated = query(client, controls, 'baseline_control')
    control = control_comparison(controls, repeated, cached['initial'], 1e-7)
    write(out / 'controls-private.json', control)
    if not control['passed']:
        raise ValueError('Exact current-input repeat failed; header ablation not executed')
    baseline_train = query(client, train, 'baseline_rich_train')
    header = query(client, records, 'baseline_original_instruction', variant=True)
    del client; gc.collect(); torch.cuda.empty_cache()
    client = load(candidate)
    trained_train = query(client, train, 'research72_rich_train')
    del client; gc.collect(); torch.cuda.empty_cache()
    for path, expected in protocol['source_sha256'].items():
        if checksum(path) != expected: raise ValueError('Read-only diagnostic source changed: ' + path)
    if bundle_identity(baseline) != baseline_id or bundle_identity(candidate) != candidate_id or queries != 304:
        raise ValueError('Read-only weight identities or exact query count changed')
    prior_train = {r['_meta']['id']: reference['rows'][r['_meta']['id']]['probabilities'] for r in train}
    training = {k: training_metrics(train, p, reference['rows']) for k, p in (
        ('original_teacher', prior_train), ('baseline_rich', baseline_train), ('research72_rich', trained_train))}
    if any(m['counts']['strict_improvement_positions'] != 19 for m in training.values()):
        raise ValueError('Fixed19 strict training improvements changed')
    summary = {'status': 'target_learning_and_original_instruction_diagnostic_complete', 'control': control,
        'training': training, 'cached_validation': cached_metrics,
        'original_instruction_validation': validation_metrics(records, header, reference['rows']),
        'cached_fourth_drift': drift_summary(records, cached, reference['rows']),
        'changed_instruction_records': sum(inference_request(r)['questions']['action']['instructions'] !=
            teachers[r['_meta']['id']]['questions']['action']['instructions'] for r in records),
        'new_model_queries': queries, 'optimizer_updates': 0, 'new_match_simulations': 0,
        'diagnostic_only': True, 'eligible_for_promotion': False, 'independent_test': False,
        'exported_or_deployed': False, 'raw_publication': False}
    write(out / 'summary-private.json', summary)
    write(out / 'status-private.json', {'status': summary['status']})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-', 'login')):
        raise RuntimeError('This diagnostic requires an approved GPU allocation')
    from tools.prepare_native_context_replay import allocation_metadata
    description = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '--oneliner'], text=True)
    allocation = allocation_metadata(description, socket.gethostname())
    print(json.dumps({'node': socket.gethostname(), 'allocation': allocation}), flush=True)
    print(json.dumps(audit(json.loads(args.config.read_text())), indent=2), flush=True)


if __name__ == '__main__': main()
