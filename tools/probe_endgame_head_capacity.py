"""TRAIN-only encoding audit and in-memory existing-head capacity probe."""
import argparse
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
    expected = {'schema': 'endgame-existing-head-capacity-v1', 'training_positions': 19,
        'encoder_queries': 19, 'encoder_frozen': True, 'existing_head_clone_only': True,
        'head_probe_registered': True, 'max_head_updates': 128, 'head_learning_rate': .001,
        'weight_decay': 0., 'gradient_clip_norm': 1., 'evaluation_every_updates': 8,
        'goal_min_training_hits': 18, 'goal_max_mean_excess_ce': .2,
        'baseline_max_probability_difference': 1e-7, 'diagnostic_only': True}
    if any(config.get(k) != v for k, v in expected.items()) or any(config.get(k) is not False
            for k in ('validation_data_used', 'candidate_training_registered', 'promotion_registered',
                      'export_registered', 'independent_confirmation', 'raw_publication')):
        raise ValueError('Only the registered TRAIN19 cloned-head capacity probe is enabled')


def encoded_public_state_audit(state, decoded, encoding):
    required = ('endgame_context', 'scores', 'gap_to_second', 'gap_to_third')
    forbidden = ('expected_terminal_utilities', 'terminal_escape4_probabilities', 'strict_original_policy_improvement')
    if (encoding.get('state_truncated') or any(k not in state or k not in decoded for k in required)
            or any(k in state or k in decoded for k in forbidden)):
        raise ValueError('Encoded public context missing/truncated or contains terminal target metadata')
    return {'required_public_fields_present': list(required), 'target_metadata_absent': True,
        'state_truncated': False, 'state_tokens': encoding['seg'].count(0), 'total_tokens': len(encoding['ids'])}


def target_stats(target, probabilities, eligible):
    keys = list(target)
    if (set(probabilities) != set(keys) or not eligible or not set(eligible) <= set(keys)
            or any(not math.isfinite(v) or v < 0 for v in [*target.values(), *probabilities.values()])
            or abs(sum(target.values()) - 1.) > 1e-5 or abs(sum(probabilities.values()) - 1.) > 1e-5):
        raise ValueError('Invalid target/probability or candidate support')
    entropy = -sum(v * math.log(v) for v in target.values() if v > 0)
    ce = -sum(v * math.log(max(probabilities[k], 1e-30)) for k, v in target.items() if v > 0)
    chosen = max(eligible, key=probabilities.get)
    return {'hit': target[chosen] > 0, 'target_ce': ce, 'target_entropy': entropy,
            'excess_ce': max(0., ce - entropy)}


def probe(config):
    validate_config(config)
    from jevmahjong.endgame.collect import digest, write
    from jevmahjong.endgame.prepare import read_jsonl, checksum
    from jevmahjong.endgame.reference import inference_request
    from jevmahjong.endgame.public_policy import action_keys
    from jevmahjong.endgame.futures import bundle_identity
    from desktop.jev_sidecar.unified_policy import identity
    from tools.audit_endgame_parameter_attribution import control_comparison
    out, run, previous = map(Path, (config['out'], config['research_data_run'], config['baseline_prediction_run']))
    if (out / 'protocol-private.json').exists() or (out / 'summary-private.json').exists():
        raise ValueError('Inspect existing capacity probe before any resubmission')
    run_config = json.loads((run / 'config-private.json').read_text())
    parent = Path(run_config['checkpoint']); parent_id = bundle_identity(parent); released = identity()
    if parent_id['identity_sha256'] not in (released['source_identity_sha256'], released['identity_sha256']):
        raise ValueError('Capacity source is not the formal v8')
    source_audit = json.loads((run / 'postfit-summary-private.json').read_text())
    if (source_audit['status'] != 'thirteenth_fit_selection_and_training_learning_audit_complete'
            or source_audit['last_step'] != 72 or source_audit['selected_step'] != 0
            or not source_audit['source_and_dataset_fingerprints_unchanged']):
        raise ValueError('Expected the completed and unqualified thirteenth data audit')
    rows = [r for r in read_jsonl(run / 'train.jsonl') if r['questions']['action']['src'] == 'endgame_terminal_fourth_improve']
    if len(rows) != 19 or len({r['_meta']['id'] for r in rows}) != 19 or any(
            r['_meta'].get('strict_original_policy_improvement') is not True for r in rows):
        raise ValueError('The fixed complete19 TRAIN-only strict source differs')
    reference = json.loads((run / 'reference-targets-private.json').read_text())
    baseline = json.loads((previous / 'baseline_rich_train-predictions-private.json').read_text())
    previous_protocol = json.loads((previous / 'protocol-private.json').read_text())
    if reference['identity_sha256'] != parent_id['identity_sha256'] or previous_protocol['baseline_identity'] != parent_id:
        raise ValueError('Reference or cached predictions use a different source model')
    input_targets = {}
    for row in rows:
        rid = row['_meta']['id']; key = digest(inference_request(row))
        if reference['rows'][rid]['request_sha256'] != key or rid not in baseline:
            raise ValueError('Actual capacity input differs from frozen source')
        target = row['questions']['action']['target']
        if key in input_targets and input_targets[key] != target:
            raise ValueError('Identical public inputs have conflicting targets; do not prune them')
        input_targets[key] = target
    upstream = Path(run_config['upstream_source']); sys.path.insert(0, str(upstream))
    paths = set(ROOT.glob('jevmahjong/**/*.py')) | set(ROOT.glob('desktop/jev_sidecar/**/*.py'))
    paths.update((upstream / 'jevany').glob('**/*.py'))
    paths.update(ROOT / name for name in ('tools/probe_endgame_head_capacity.py',
        'tools/audit_endgame_parameter_attribution.py', 'tools/prepare_native_context_replay.py',
        'configs/training/endgame-existing-head-capacity-v1.json'))
    paths.update(run / name for name in ('train.jsonl', 'config-private.json', 'reference-targets-private.json', 'postfit-summary-private.json'))
    paths.update(previous / name for name in ('baseline_rich_train-predictions-private.json', 'protocol-private.json'))
    protocol = {'configuration': config, 'parent_identity': parent_id,
        'record_ids': [r['_meta']['id'] for r in rows], 'unique_public_inputs': len(input_targets),
        'source_sha256': {str(p.resolve()): checksum(p) for p in sorted(paths)},
        'versions': {name: importlib.metadata.version(name) for name in ('torch', 'transformers', 'peft', 'riichienv', 'mahjong')},
        'validation_data_used': False, 'eligible_for_promotion': False}
    write(out / 'protocol-private.json', protocol)
    import torch
    from jevmahjong.policy.routing import load_model
    from jevany.api import validate_distribution
    from jevany.model import tokenizer_of
    torch.manual_seed(2026111914)
    client = load_model(parent); client.runtime.model.eval().requires_grad_(False)
    if client.runtime.inference_options.prefix_cache_size != 0:
        raise ValueError('Capture requires disabled inference prefix cache')
    original_head = client.runtime.model.head
    if original_head.residual is not None or original_head.temperature != 1.:
        raise ValueError('Expected the unchanged plain temperature1 pointer head')
    original_parameter_ids = {id(p) for p in client.runtime.model.parameters()}
    head = copy.deepcopy(original_head).requires_grad_(True)
    initial = {k: v.detach().cpu().clone() for k, v in original_head.state_dict().items()}
    if any(not torch.equal(v.detach().cpu(), initial[k]) for k, v in head.state_dict().items()):
        raise ValueError('Probe head is not an exact clone of the existing head')
    if sum(p.numel() for p in head.parameters()) != 524800:
        raise ValueError('Probe must retain exactly the existing524800 head parameters')
    if any(p.requires_grad for p in client.runtime.model.parameters()):
        raise ValueError('Original encoder and head must remain frozen')
    captured, encodings = [], []
    original_admit = client.runtime._admit
    def traced_admit(record):
        value = original_admit(record); enc = value[1]; count = enc['seg'].count(0)
        decoded = tokenizer_of(client.runtime.tok).decode(enc['ids'][:count], skip_special_tokens=False)
        encodings.append(encoded_public_state_audit(record['state'], decoded, enc))
        return value
    def capture(module, args):
        if len(args) != 2 or args[0].ndim != 1 or args[1].ndim != 2:
            raise ValueError('Unexpected pointer head call shape')
        with torch.inference_mode(False):
            captured.append({'decide': args[0].detach().to('cpu', torch.float32).clone(),
                'options': args[1].detach().to('cpu', torch.float32).clone(),
                'autocast_enabled': torch.is_autocast_enabled('cuda'),
                'autocast_dtype': str(torch.get_autocast_dtype('cuda')).split('.')[-1]})
    client.runtime._admit = traced_admit
    handle = original_head.register_forward_pre_hook(capture)
    features, actual, token_audits = {}, {}, {}
    try:
        for row in rows:
            captured.clear(); encodings.clear(); rid = row['_meta']['id']
            answer = client.system_one(**inference_request(row)); keys = list(row['questions']['action']['criteria'])
            probabilities, _ = validate_distribution(answer['answers']['action']['probabilities'], keys)
            if len(captured) != 1 or len(encodings) != 1 or captured[0]['options'].shape[0] != len(keys):
                raise ValueError('A single question must have exactly one matching head/encoding call')
            features[rid] = dict(captured[0], keys=keys)
            token_audits[rid] = encodings[0]; actual[rid] = dict(zip(keys, probabilities))
    finally:
        handle.remove(); client.runtime._admit = original_admit
    repeated = control_comparison(rows, actual, baseline, 1e-7)
    write(out / 'end_to_end-control-private.json', repeated)
    write(out / 'token-audits-private.json', token_audits)
    if not repeated['passed']: raise ValueError('Formal-v8 capture differs from cached predictions')
    torch.save({'features': features, 'parent_identity': parent_id, 'diagnostic_only': True}, out / 'frozen-features-private.pt')
    for value in features.values():
        value['decide'] = value['decide'].to('cuda').clone()
        value['options'] = value['options'].to('cuda').clone()
    def logits(rid):
        value = features[rid]
        with torch.autocast('cuda', enabled=value['autocast_enabled'], dtype=getattr(torch, value['autocast_dtype'])):
            return head(value['decide'], value['options']).float()
    def predictions():
        result = {}; head.eval()
        with torch.no_grad():
            for row in rows:
                rid = row['_meta']['id']; keys = features[rid]['keys']
                p, _ = validate_distribution(logits(rid).softmax(-1).cpu().tolist(), keys)
                result[rid] = dict(zip(keys, p))
        return result
    replayed = predictions(); replay_control = control_comparison(rows, replayed, actual, 1e-7)
    write(out / 'head-replay-control-private.json', replay_control)
    if not replay_control['passed']: raise ValueError('Captured head features do not reproduce real inference')
    del client, original_admit, original_head; gc.collect(); torch.cuda.empty_cache()
    optimizer = torch.optim.AdamW(head.parameters(), lr=.001, weight_decay=0.)
    if {id(p) for group in optimizer.param_groups for p in group['params']} & original_parameter_ids:
        raise ValueError('Optimizer must not contain any original model parameter')
    targets = {r['_meta']['id']: torch.tensor([r['questions']['action']['target'][k]
        for k in features[r['_meta']['id']]['keys']], dtype=torch.float32, device='cuda') for r in rows}
    curve = []
    def evaluate(step, p):
        stats = [target_stats(r['questions']['action']['target'], p[r['_meta']['id']], action_keys(r)) for r in rows]
        value = {'step': step, 'positions': 19, 'target_hits': sum(s['hit'] for s in stats),
            'mean_target_ce': sum(s['target_ce'] for s in stats) / 19,
            'mean_target_entropy': sum(s['target_entropy'] for s in stats) / 19,
            'mean_excess_ce': sum(s['excess_ce'] for s in stats) / 19,
            'validation_data_used': False}
        curve.append(value); write(out / 'curve-private.json', curve); write(out / 'status-private.json', value)
        print(json.dumps(value), flush=True); return value
    current = evaluate(0, replayed)
    for step in range(1, 129):
        head.train(); optimizer.zero_grad(set_to_none=True)
        loss = sum(-(targets[r['_meta']['id']] * logits(r['_meta']['id']).log_softmax(-1)).sum() for r in rows) / 19
        if not torch.isfinite(loss): raise ValueError('Nonfinite probe objective')
        loss.backward(); norm = torch.nn.utils.clip_grad_norm_(head.parameters(), 1.)
        if not torch.isfinite(norm): raise ValueError('Nonfinite probe gradient')
        optimizer.step()
        if step % 8 == 0:
            current = evaluate(step, predictions())
            if current['target_hits'] >= 18 and current['mean_excess_ce'] <= .2: break
    for path, expected in protocol['source_sha256'].items():
        if checksum(path) != expected: raise ValueError('Read-only source changed during capacity probe')
    if bundle_identity(parent) != parent_id: raise ValueError('Original v8 files changed')
    final = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    relative_change = math.sqrt(sum(((v - initial[k]) ** 2).sum().item() for k, v in final.items()) /
                                sum((v ** 2).sum().item() for v in initial.values()))
    torch.save({'head': final, 'diagnostic_only': True, 'parent_identity': parent_id}, out / 'probe-head-private.pt')
    summary = {'status': 'existing_head_capacity_probe_complete', 'end_to_end_control': repeated,
        'head_replay_control': replay_control, 'encoded_positions': 19, 'all_public_context_checks_passed': True,
        'initial': curve[0], 'final': curve[-1], 'goal_reached': current['target_hits'] >= 18 and current['mean_excess_ce'] <= .2,
        'relative_head_parameter_l2_change': relative_change, 'trainable_head_parameters': 524800,
        'original_encoder_optimizer_parameters': 0, 'full_encoder_queries': 19, 'head_updates': curve[-1]['step'],
        'original_weight_files_unchanged': True, 'source_fingerprints_unchanged': True,
        'validation_data_used': False, 'new_simulations': 0, 'candidate_training_registered': False,
        'eligible_for_promotion': False, 'exported_or_deployed': False, 'raw_publication': False}
    write(out / 'summary-private.json', summary); write(out / 'status-private.json', {'status': summary['status']})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-', 'login')):
        raise RuntimeError('Capacity probe requires a scheduled GPU allocation')
    from tools.prepare_native_context_replay import allocation_metadata
    description = subprocess.check_output(['scontrol', 'show', 'job', os.environ['SLURM_JOB_ID'], '--oneliner'], text=True)
    allocation = allocation_metadata(description, socket.gethostname())
    print(json.dumps({'node': socket.gethostname(), 'allocation': allocation}), flush=True)
    print(json.dumps(probe(json.loads(args.config.read_text())), indent=2), flush=True)


if __name__ == '__main__': main()
