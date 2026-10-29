"""Incremental point-ranking collection, stopping and predeclared positive selection."""
import copy
import hashlib
import json
import math
from pathlib import Path
import shutil
import tempfile

import torch
from head_rl import HeadPolicy, fingerprint, summarize, write
from single_decision_rl import SingleDecisionPolicy
from scripts.audit_score_confirmation import audit_rows

ROOT = Path(__file__).resolve().parent
ORIGINAL_HEAD = '3c586e1bf6e86892c29a1f76bbab4fda4f714b17a31f87c6fe871157748a7d5b'
SELECTION_POLICY = 'best_positive_complete_development'


class IncrementalFrozenTeacherPolicy(SingleDecisionPolicy):
    """Multiple fresh data batches from one fixed teacher; ranking is off-policy."""
    def __init__(self, config, checkpoint=None):
        if checkpoint is not None or config['training_method'] != 'point_delta_weighted_reference_relative_pairwise':
            raise ValueError('Incremental collection requires a fresh point-ranking optimizer')
        if config['advantage_estimator'] != 'greedy_reference' or config['reward_mode'] != 'net_points':
            raise ValueError('Incremental collection requires signed paired net points')
        if config.get('teacher_control_stationary_across_stages') is not True:
            raise ValueError('The teacher must stay frozen across fresh-data batches')
        if fingerprint(ROOT/config['reference']/'head.pt') != config['initial_reference_head_sha256']:
            raise ValueError('Prepared training initialization changed')
        HeadPolicy.__init__(self, config)
        if any(not torch.equal(v, self.reference.state_dict()[k]) for k,v in self.head.state_dict().items()):
            raise ValueError('Initial actor must equal the frozen collection teacher')
        self.controls = {}


def incremental_gate(check, candidate, baseline, config, paired=None):
    gain = candidate['mean_net_points'] - baseline['mean_net_points']
    interval = paired.get('primary_delta_ci95') if paired else None
    if interval is None or len(interval) != 2 or not all(math.isfinite(x) for x in interval):
        raise ValueError('Incremental stopping requires a finite paired development interval')
    changes = {'net_points_gain': gain, 'paired_net_points_ci95': interval,
               'desired_gain_only': config['desired_net_points_gain'], 'fixed_gain_gate_enabled': False}
    if gain <= -config['net_points_drop_stop'] or interval[1] < 0:
        return 'stop_regression', changes
    if check >= config['max_validation_checks']:
        return 'stop_bounded_data_phase', changes
    return 'continue', changes


def choose_positive(results, baseline_mean):
    eligible = [r for r in results if math.isfinite(r['metrics']['mean_net_points'])
                and r['metrics']['mean_net_points'] >= 0
                and r['metrics']['mean_net_points'] > baseline_mean]
    return max(eligible, key=lambda r:(r['metrics']['mean_net_points'], -r['update'])) if eligible else None


def verified_development(source):
    protocol = json.loads((source/'protocol.json').read_text())
    config = protocol['config']
    if hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest() != protocol['sha256']:
        raise ValueError('Source protocol changed')
    for name, expected in protocol['code_sha256'].items():
        if fingerprint(ROOT/name) != expected:
            raise ValueError(f'Source numerical code changed: {name}')
    analysis = json.loads((source/'pilot-analysis.json').read_text())
    if analysis['status'] != 'pilot_complete':
        raise ValueError('Wait for the bounded source run to complete')
    audit_config = {**config, 'confirmation_seed_start': config['development_seed_start'],
                    'confirmation_seeds': config['development_seeds']}
    def native_rows(path):
        rows = [json.loads(s) for s in path.read_text().splitlines()]
        audit_rows(rows, audit_config)
        return summarize(rows)
    baseline = native_rows(source/'development-baseline/games.jsonl')
    results = json.loads((source/'development-results.json').read_text())
    if analysis['development'] != results:
        raise ValueError('Analysis and development records differ')
    for entry in results:
        report = native_rows(source/f"development-update-{entry['update']:06d}"/'games.jsonl')
        if report != entry['metrics']:
            raise ValueError('Development summary differs from complete native matches')
        checkpoint = ROOT/entry['checkpoint']
        if not checkpoint.resolve().is_relative_to((source/'checkpoints').resolve()):
            raise ValueError('Checkpoint lies outside its source run')
        complete = json.loads((checkpoint/'complete.json').read_text())
        if complete['head_sha256'] != fingerprint(checkpoint/'head.pt') or complete['update'] != entry['update']:
            raise ValueError('Checkpoint completion fingerprint differs')
    return protocol, baseline, results


def select_incremental(source):
    protocol, baseline, results = verified_development(source)
    if protocol['config'].get('selection_policy') != SELECTION_POLICY:
        raise ValueError('Positive selection must be declared before this training run')
    chosen = choose_positive(results, baseline['mean_net_points'])
    record = {'selection_policy': SELECTION_POLICY, 'source_run': str(source),
              'protocol_sha256': protocol['sha256'], 'baseline_mean_net_points': baseline['mean_net_points'],
              'selected': chosen, 'fixed_gain_gate_enabled': False}
    if chosen:
        record['candidate_head_sha256'] = fingerprint(ROOT/chosen['checkpoint']/'head.pt')
    write(source/'incremental-selection.json', record)
    return record


def confirmation_config(config):
    c = copy.deepcopy(config)
    if c.get('selection_policy') != SELECTION_POLICY or c.get('confirmation_qualification') != 'positive_paired_ci95':
        raise ValueError('The new positive-evidence rule must be declared before data collection')
    if c.get('confirmation_min_ci95_lower') != 0 or c['confirmation_seeds'] < 128:
        raise ValueError('Keep at least128 independent streams and strict positive lower interval')
    if c['game_mode'] != '4p-red-east' or c['reward_mode'] != 'net_points' or c['seed_scheme'] != 'spaced-v1':
        raise ValueError('Keep the complete East net-point benchmark')
    if c['final_benchmark_head_sha256'] != ORIGINAL_HEAD:
        raise ValueError('Final benchmark must remain the original2000-step model')
    c.update(reference=c['final_benchmark_reference'], target_net_points_gain=0., target_mean_net_points=0.)
    return c


def prepare_positive_confirmation(source, out):
    if out.exists():
        raise ValueError('Preserve the existing independent study')
    protocol, baseline, results = verified_development(source)
    recorded = json.loads((source/'incremental-selection.json').read_text())
    chosen = choose_positive(results, baseline['mean_net_points'])
    if not chosen or recorded['selected'] != chosen or recorded['protocol_sha256'] != protocol['sha256']:
        raise ValueError('Missing or changed predeclared positive candidate selection')
    config = confirmation_config(protocol['config'])
    checkpoint, reference = ROOT/chosen['checkpoint'], ROOT/config['reference']
    head_sha = fingerprint(checkpoint/'head.pt')
    if head_sha != recorded['candidate_head_sha256'] or fingerprint(reference/'head.pt') != ORIGINAL_HEAD:
        raise ValueError('Selected candidate or original benchmark changed')
    for name in ['adapter_config.json','adapter_model.safetensors','tokenizer.json','tokenizer_config.json','chat_template.jinja']:
        if fingerprint(checkpoint/name) != fingerprint(reference/name):
            raise ValueError(f'Frozen input differs from original benchmark: {name}')
    files = sorted(source.glob('training-games-stage-*.json'))
    training = {g['seed'] for p in files for g in json.loads(p.read_text())}
    development = set(range(config['development_seed_start'], config['development_seed_start']+config['development_seeds']))
    confirmation = set(range(config['confirmation_seed_start'], config['confirmation_seed_start']+config['confirmation_seeds']))
    if not training or training & development or training & confirmation or development & confirmation:
        raise ValueError('Missing provenance or overlapping training/evaluation streams')
    training_reference = protocol['config']['reference']
    config['out'] = str(out)
    frozen = {**protocol, 'config':config,
              'sha256':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
              'reference_head_sha256': ORIGINAL_HEAD,
              'reference_adapter_sha256':fingerprint(reference/'adapter_model.safetensors'),
              'source_training_reference':training_reference, 'source_run':str(source),
              'study_type':'positive_paired_evidence_after_user_incremental_policy'}
    selection = {'checkpoint':chosen['checkpoint'], 'candidate_head_sha256':head_sha,
                 'development_update':chosen['update'], 'config':config,
                 'selection_rule':SELECTION_POLICY, 'source_training_reference':training_reference,
                 'source_run':str(source)}
    out.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent,prefix=out.name+'.prepare-') as temporary:
        staged=Path(temporary)/'study';staged.mkdir()
        for name,value in [('protocol.json',frozen),('confirmation-selection.json',selection),
                           ('status.json',{'status':'awaiting_confirmation','checkpoint':chosen['checkpoint']})]:
            write(staged/name,value)
        for p in files+sorted(source.glob('reference-controls-stage-*.json')):
            shutil.copy2(p,staged/p.name)
        for name,newname in [('protocol.json','source-training-protocol.json'),
                             ('pilot-analysis.json','source-training-analysis.json'),
                             ('incremental-selection.json','source-incremental-selection.json')]:
            shutil.copy2(source/name,staged/newname)
        staged.rename(out)
    return selection
