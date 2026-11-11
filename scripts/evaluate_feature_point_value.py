"""Frozen point predictor versus preserved stage2 hybrid on registered fresh walls."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from call_policy import digest, request_hash
from clean_policy_evaluation import aggregate, SUITES, POLICIES
from encoder_call_policy import EncoderCallRouter
from feature_point_value import FeaturePointClient, point_features
from guide_call_value import utility_preferences
from head_rl import write
from scripts.collect_feature_points import assert_frozen, validate_wall
from scripts.evaluate_call_repair import Agent
from scripts.evaluate_guide_calls import game
from scripts.train_call_repair import make_head
from stage2_ablation import FullStage2Client

CODE = ['feature_point_value.py', 'stage2_ablation.py',
    'scripts/evaluate_feature_point_value.py', 'scripts/audit_feature_point_value.py',
    'scripts/evaluate_guide_calls.py', 'scripts/evaluate_call_repair.py',
    'scripts/train_call_repair.py', 'scripts/collect_feature_points.py',
    'scripts/fit_feature_points.py', 'clean_policy_evaluation.py', 'guide_development_audit.py',
    'guide_call_value.py', 'guide_call_features.py', 'diagonal_value.py', 'head_rl.py',
    'encoder_call_policy.py', 'call_option_features.py', 'clean_policy_data.py',
    'call_policy.py', 'semantic.py', 'arena.py', 'scoring_audit.py', 'seed_control.py',
    'checkpoint_identity.py', 'vendor/JevAny/jevany/model.py',
    'vendor/JevAny/jevany/inference.py', 'vendor/JevAny/jevany/data.py',
    'vendor/JevAny/jevany/client.py']


def verify_protocol(protocol):
    cfg = protocol['config']; source_cfg = json.loads((ROOT / cfg['source_config']).read_text())
    if source_cfg != protocol['training_config']: raise ValueError('Point training config changed')
    run = ROOT / source_cfg['out']
    source = json.loads((run / 'protocol.json').read_text())
    if source['config'] != source_cfg: raise ValueError('Collection protocol differs')
    assert_frozen(source, source_cfg)
    if digest(run / 'fit-result.json') != protocol['fit_result_sha256']:
        raise ValueError('Point fit selection changed')
    if digest(run / 'point-value-private.pt') != protocol['point_checkpoint_sha256']:
        raise ValueError('Point predictor changed')
    for name, expected in protocol['code_sha256'].items():
        if digest(ROOT / name) != expected: raise ValueError('Point evaluation source changed: ' + name)


def native_parity(router, cpu_head, run, cfg, source_cfg):
    checked = 0; seen = set(); max_value_error = 0.; max_teacher_error = 0.
    for shard in sorted((run / 'walls-private').glob('*.pt')):
        payload = torch.load(shard, map_location='cpu', weights_only=False)
        validate_wall(payload, payload['seed'], source_cfg)
        for row in payload['rows']:
            request = row['public_request']; key = request_hash(request)
            if key in seen: continue
            seen.add(key); result = router.system_one(**request); captured = router.last
            error = float((captured['teacher_probabilities'] - row['reference']).abs().max())
            max_teacher_error = max(max_teacher_error, error)
            if error > 2e-5 or int(row['reference'].argmax()) != captured['teacher_index']:
                raise ValueError('Served stage2 teacher differs from frozen data collection')
            matrix = point_features(cpu_head, captured['decide'], captured['options'], request)
            relative = (matrix.float() - matrix.float()[captured['teacher_index']]).double()
            values = relative @ router.coefficient.cpu()
            value_error = float((values - captured['values']).abs().max()); max_value_error = max(max_value_error, value_error)
            if value_error > cfg['point_value_parity_atol']:
                raise ValueError('CPU/native action differences disagree')
            _, chosen = utility_preferences(values, captured['teacher_index'])
            keys = list(request['questions']['action']['criteria'])
            p = result['answers']['action']['probabilities']
            if chosen != captured['chosen'] or keys[chosen] != max(keys, key=lambda k: p[k]):
                raise ValueError('CPU/native point decisions disagree')
            if float(captured['values'][captured['teacher_index']]) != 0.:
                raise ValueError('Teacher action must have exactly zero relative utility')
            checked += 1
            if checked >= cfg['native_parity_requests']: break
        if checked >= cfg['native_parity_requests']: break
    if checked != cfg['native_parity_requests']: raise ValueError('Insufficient distinct native TRAIN calls')
    parent = json.loads((ROOT / source_cfg['teacher_source_config']).read_text())
    prepared = json.loads((ROOT / parent['out'] / 'preparation.json').read_text())
    data = ROOT / parent['data'] / 'train.jsonl'
    if digest(data) != prepared['files']['train']: raise ValueError('Ordinary parity TRAIN changed')
    from jevany.data import api_request
    ordinary = 0
    for line in data.open():
        record = json.loads(line)
        if record['_meta']['phase'] == 'reaction': continue
        request = api_request(record)
        before = router.ordinary.system_one(**request)['answers']['action']['probabilities']
        after = router.system_one(**request)['answers']['action']['probabilities']
        if before != after: raise ValueError('Point predictor changed ordinary decisions')
        ordinary += 1
        if ordinary == 6: break
    if ordinary != 6: raise ValueError('Ordinary route coverage incomplete')
    return {'distinct_train_calls': checked, 'ordinary_requests_unchanged': ordinary,
        'maximum_cpu_native_value_error': max_value_error, 'maximum_teacher_probability_error': max_teacher_error,
        'cpu_native_decisions_equal': True, 'teacher_relative_value_exactly_zero': True,
        'human_test_opened': False, 'source': 'TRAIN-only point traces and already-opened ordinary TRAIN requests'}


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = (out / 'evaluation.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source_cfg = json.loads((ROOT / cfg['source_config']).read_text()); run = ROOT / source_cfg['out']
    source = json.loads((run / 'protocol.json').read_text())
    if source_cfg != source['config']: raise ValueError('Source training registration changed')
    assert_frozen(source, source_cfg)
    fit = json.loads((run / 'fit-result.json').read_text())
    if cfg['stage'] not in ('development', 'independent_confirmation'): raise ValueError('Unknown evaluation stage')
    prefix = 'development' if cfg['stage'] == 'development' else 'confirmation'
    if (cfg['seed_start'] != source_cfg[prefix + '_seed_start']
            or cfg['seeds'] != source_cfg[prefix + '_seeds']):
        raise ValueError('Unregistered fresh point-evaluation walls')
    if fit['status'] != 'point_predictor_fit':
        write(out / 'evaluation-skipped.json', {'reason': fit['status'], 'deployed': False}); return
    if digest(run / 'point-value-private.pt') != fit['checkpoint_sha256']:
        raise ValueError('Fitted point checkpoint changed')
    if cfg['stage'] == 'independent_confirmation':
        dev = ROOT / cfg['development_out']
        audit_path = dev / 'evaluation-audit.json'
        if (dev / 'evaluation-skipped.json').exists():
            write(out / 'evaluation-skipped.json', {'reason': 'development_skipped', 'deployed': False}); return
        audit = json.loads(audit_path.read_text())
        if audit['status'] != 'passed_complete': raise ValueError('Complete development audit required')
        if audit['overall_paired_gain']['gain'] <= 0:
            write(out / 'evaluation-skipped.json', {'reason': 'nonpositive_parent_development', 'deployed': False}); return
        dev_protocol = json.loads((dev / 'protocol.json').read_text())
        if (dev_protocol['point_checkpoint_sha256'] != fit['checkpoint_sha256']
                or dev_protocol['fit_result_sha256'] != digest(run / 'fit-result.json')):
            raise ValueError('Confirmation candidate differs from development')
    protocol = {'config': cfg, 'training_config': source_cfg,
        'point_checkpoint_sha256': fit['checkpoint_sha256'], 'fit_result_sha256': digest(run / 'fit-result.json'),
        'teacher_files': source['teacher_files'], 'ordinary_files': source['reference_files'],
        'code_sha256': {n: digest(ROOT / n) for n in CODE}, 'candidate_frozen': True,
        'arms': {'reference': 'preserved stage2 chi/pon/pass with oldB ordinary',
            'candidate': 'paired-terminal-point utilities with same stage2 features and oldB ordinary'},
        'primary': 'candidate minus preserved stage2 hybrid complete-East terminal net points, whole-wall CI',
        'human_test_opened': False, 'deployed': False}
    target = out / 'protocol.json'
    if target.exists() and json.loads(target.read_text()) != protocol: raise ValueError('Resume point protocol changed')
    if not target.exists(): write(target, protocol)
    verify_protocol(protocol)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False); torch.backends.cuda.enable_mem_efficient_sdp(False)
    from jevany import JevModel
    from jevany.checkpoint import LoadOptions
    from jevany.inference import InferenceOptions
    base = Path(os.environ.get('JEV_BASE_MODEL_DIR', Path.home() /
        '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
    def load(path):
        return JevModel.from_pretrained(str(ROOT / path), device='cuda', dtype='bf16',
            options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
                fused_kernels=False, base_load_path=str(base)),
            inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192,
                max_packed_tokens=8192, prefix_cache_size=0))
    write(out / 'status.json', {'status': 'loading_point_candidate_and_stage2_teacher', 'games': 0})
    ordinary = load(source_cfg['reference']); teacher = load(source_cfg['teacher_checkpoint'])
    point = torch.load(run / 'point-value-private.pt', map_location='cpu', weights_only=True)
    if point['teacher_files'] != source['teacher_files'] or point['reference_files'] != source['reference_files']:
        raise ValueError('Point checkpoint representations have different identities')
    router = FeaturePointClient(ordinary, teacher, point)
    cpu_head = make_head(torch.load(ROOT / source_cfg['teacher_checkpoint'] / 'head.pt',
        map_location='cpu', weights_only=True), 'cpu').eval().requires_grad_(False)
    try:
        write(out / 'native-parity.json', native_parity(router, cpu_head, run, cfg, source_cfg))
        agents = {'reference': Agent(EncoderCallRouter(ordinary, FullStage2Client(teacher))), 'candidate': Agent(router)}
        strong = Agent(ordinary); raw = out / 'games-private.jsonl'
        rows = [json.loads(l) for l in raw.read_text().splitlines()] if raw.exists() else []
        done = {(r['suite'], r['seed'], r['seat'], r['policy']) for r in rows}
        seeds = range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds'])
        expected = {(s, n, p, k) for s in SUITES for n in seeds for p in range(4) for k in POLICIES}
        if len(done) != len(rows) or not done <= expected: raise ValueError('Foreign or duplicate resumed point game')
        started = time.monotonic()
        with raw.open('a') as stream:
            for seed in seeds:
                for suite in SUITES:
                    for seat in range(4):
                        for name in POLICIES:
                            if (suite, seed, seat, name) in done: continue
                            row = {'policy': name, **game(agents[name], strong, seed, seat, suite)}
                            stream.write(json.dumps(row) + '\n'); stream.flush(); rows.append(row)
                            status = {'status': 'point_value_vs_stage2_games', 'games': len(rows),
                                'total_games': len(expected), 'stage': cfg['stage'], 'seconds': time.monotonic() - started}
                            write(out / 'status.json', status)
                            if len(rows) % 24 == 0: print(json.dumps(status), flush=True)
        verify_protocol(protocol); write(out / 'summary.json', aggregate(rows, cfg))
        write(out / 'status.json', {'status': 'point_games_complete_audit_pending', 'games': len(rows)})
    finally: router.close()


if __name__ == '__main__': main()
