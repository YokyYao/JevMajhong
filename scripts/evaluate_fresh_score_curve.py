"""Evaluate fresh-match continuation checkpoints with the frozen parent once."""
import argparse
import fcntl
import gc
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files
from clean_policy_evaluation import aggregate, SUITES
from encoder_call_policy import EncoderCallRouter
from feature_continuation import verify_preparation
from fresh_score_curve import policy_name, summarize
from head_rl import write
from scripts.evaluate_call_repair import Agent
from scripts.evaluate_guide_calls import game
from scripts.evaluate_stage2_ablation import parity
from stage2_ablation import FullStage2Client

CODE = ['fresh_score_curve.py', 'scripts/evaluate_fresh_score_curve.py',
    'scripts/audit_fresh_score_curve.py', 'scripts/evaluate_stage2_ablation.py',
    'stage2_ablation.py', 'feature_continuation.py', 'feature_training_guard.py',
    'checkpoint_identity.py', 'scripts/evaluate_guide_calls.py', 'scripts/evaluate_call_repair.py',
    'scripts/train_call_repair.py', 'clean_policy_evaluation.py', 'guide_development_audit.py',
    'encoder_call_policy.py', 'call_option_features.py', 'guide_call_features.py',
    'clean_policy_data.py', 'call_policy.py', 'semantic.py', 'arena.py', 'scoring_audit.py',
    'seed_control.py', 'vendor/JevAny/jevany/inference.py', 'vendor/JevAny/jevany/data.py',
    'vendor/JevAny/jevany/model.py', 'vendor/JevAny/jevany/client.py']


def registration(cfg):
    if (cfg['scope'] != 'calls' or cfg['game_mode'] != '4p-red-east'
            or cfg['stage'] not in ('development', 'independent_confirmation')):
        raise ValueError('Registered calls-only continuation score comparison required')
    source = json.loads((ROOT / cfg['source_config']).read_text()); run = ROOT / source['out']
    preparation = json.loads((run / 'preparation.json').read_text())
    verify_preparation(ROOT, source, preparation)
    training = json.loads((run / 'protocol.json').read_text())
    result = json.loads((run / 'result.json').read_text())
    if training['config'] != source or result['parent_checkpoint'] != source['warm_start']:
        raise ValueError('Continuation source/parent changed')
    if result['status'] not in ('clean_policy_candidate', 'clean_policy_no_admissible_improvement'):
        raise ValueError('Training finalization incomplete')
    metrics = json.loads((run / 'training-run/training_metrics.json').read_text())
    if (result['optimizer_steps'] != metrics['optimizer_steps'] or metrics['rejected_records']
            or metrics['admitted_records'] != preparation['records']['train']):
        raise ValueError('Actual training exposure or admission differs')
    initialized = json.loads((run / 'initialization-audit.json').read_text())
    if not initialized['warm_start_exact_adapter_and_head'] or not initialized['no_oldB_weights_loaded']:
        raise ValueError('Exact stage2 initialization not verified')
    basis = json.loads((ROOT / cfg['scope_evidence']).read_text())
    if basis['status'] != 'passed_complete' or basis['primary_full_minus_hybrid']['gain'] > 0:
        raise ValueError('Calls-only scope must match the completed full-policy ablation decision')
    prefix = 'development' if cfg['stage'] == 'development' else 'confirmation'
    if cfg['seed_start'] != source[prefix + '_seed_start'] or cfg['seeds'] != source[prefix + '_seeds']:
        raise ValueError('Unregistered continuation score walls')
    planned = set(source['score_checkpoints'])
    if cfg['include_training_selected'] and result['selected_step']:
        planned.add(result['selected_step'])
    available, missing = {}, []
    for step in sorted(planned):
        if not isinstance(step, int) or not 0 < step <= source['max_steps']:
            raise ValueError('Invalid checkpoint step')
        if step > result['optimizer_steps']:
            missing.append(step); continue
        path = run / 'training-run/checkpoints' / f'step-{step:06d}'
        available[step] = {'checkpoint': str(path.relative_to(ROOT)),
            'files': completed_checkpoint_files(path, step)}
    if not available: raise ValueError('No completed updated checkpoint to score')
    parent_files = completed_checkpoint_files(ROOT / source['warm_start'], source['parent_step'])
    source_files = {cfg['source_config']: digest(ROOT / cfg['source_config']),
        cfg['scope_evidence']: digest(ROOT / cfg['scope_evidence'])}
    for name in ('protocol.json', 'result.json', 'preparation.json', 'initialization-audit.json',
                 'token-admission.json', 'training-run/training_metrics.json'):
        path = run / name; source_files[str(path.relative_to(ROOT))] = digest(path)
    for step in available:
        path = run / f'training-run/training_eval/step-{step:06d}/development/rows.json'
        source_files[str(path.relative_to(ROOT))] = digest(path)
    parent_cfg = json.loads((ROOT / source['parent_config']).read_text())
    parent_rows_path = ROOT / parent_cfg['out'] / f"training-run/training_eval/step-{source['parent_step']:06d}/development/rows.json"
    baseline_path = run / 'training-run/training_eval/step-000000/development/rows.json'
    old = {r['id']: r for r in json.loads(parent_rows_path.read_text())}
    new = {r['id']: r for r in json.loads(baseline_path.read_text())}
    if set(old) != set(new) or len(old) != preparation['records']['validation']:
        raise ValueError('Initial parent validation coverage differs')
    for key, row in new.items():
        previous = old[key]
        if (row['keys'] != previous['keys'] or len(row['p']) != len(previous['p'])
                or max(abs(a - b) for a, b in zip(row['p'], previous['p'])) > 2e-5
                or max(range(len(row['p'])), key=lambda i: row['p'][i]) !=
                   max(range(len(previous['p'])), key=lambda i: previous['p'][i])):
            raise ValueError('Initial candidate probabilities differ from the exact parent')
    for path in (parent_rows_path, baseline_path):
        source_files[str(path.relative_to(ROOT))] = digest(path)
    return source, available, missing, parent_files, source_files


def verify_protocol(protocol):
    source, available, missing, parent_files, source_files = registration(protocol['config'])
    # JSON object keys are strings, while registration uses integer steps.
    normalized = {str(s): v for s, v in available.items()}
    if (source != protocol['training_config'] or normalized != protocol['checkpoints']
            or missing != protocol['missing_due_to_early_stop'] or parent_files != protocol['parent_files']
            or source_files != protocol['source_files']):
        raise ValueError('Continuation evaluation registration changed')
    for name, expected in protocol['code_sha256'].items():
        if digest(ROOT / name) != expected: raise ValueError('Score-curve source changed: ' + name)
    for name, expected in protocol['ordinary_files'].items():
        if digest(ROOT / source['reference'] / name) != expected: raise ValueError('Ordinary oldB changed')
    for name, expected in protocol.get('confirmation_files', {}).items():
        if digest(ROOT / name) != expected: raise ValueError('Confirmation development result changed')


def main():
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--config', type=Path, required=True)
    p.add_argument('--resume', action='store_true'); args = p.parse_args()
    cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = (out / 'evaluation.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source, available, missing, parent_files, source_files = registration(cfg)
    training = json.loads((ROOT / source['out'] / 'protocol.json').read_text())
    dev_protocol = None; selected_step = None; confirmation_files = {}
    if cfg['stage'] == 'independent_confirmation':
        dev = ROOT / cfg['development_out']
        audit = json.loads((dev / 'evaluation-audit.json').read_text())
        if audit['status'] != 'passed_complete': raise ValueError('Complete score-curve development audit required')
        selected_step = audit['confirmation_step']
        if selected_step is None:
            write(out / 'evaluation-skipped.json', {'reason': 'no_positive_parent_gain', 'deployed': False}); return
        dev_protocol = json.loads((dev / 'protocol.json').read_text()); verify_protocol(dev_protocol)
        if (dev_protocol['checkpoints'][str(selected_step)] != available[selected_step]
                or dev_protocol['parent_files'] != parent_files or dev_protocol['config']['scope'] != cfg['scope']):
            raise ValueError('Confirmation must keep exact development candidate/parent/scope')
        confirmation_files = {str((dev / n).relative_to(ROOT)): digest(dev / n)
            for n in ('protocol.json', 'evaluation-audit.json', 'summary.json')}
    protocol = {'config': cfg, 'training_config': source,
        'checkpoints': {str(s): v for s, v in available.items()},
        'missing_due_to_early_stop': missing, 'parent_checkpoint': source['warm_start'],
        'parent_files': parent_files, 'ordinary_files': training['reference_files'],
        'source_files': source_files, 'code_sha256': {n: digest(ROOT / n) for n in CODE},
        'selected_confirmation_step': selected_step,
        'confirmation_files': confirmation_files,
        'primary': 'complete-East paired terminal net points versus preserved stage2 hybrid; whole-wall CI',
        'scope': 'oldB ordinary/rare in all arms; only chi/pon/pass checkpoint changes',
        'parent_run_once': True, 'human_test_opened': False, 'deployed': False}
    target = out / 'protocol.json'
    if target.exists() and json.loads(target.read_text()) != protocol: raise ValueError('Resume score-curve protocol differs')
    if not target.exists(): write(target, protocol)
    verify_protocol(protocol)
    steps = [selected_step] if selected_step is not None else sorted(available)
    names = ['parent', *(policy_name(s) for s in steps)]
    seeds = range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds'])
    expected = {(s, n, p, name) for s in SUITES for n in seeds for p in range(4) for name in names}
    raw = out / 'games-private.jsonl'
    rows = [json.loads(l) for l in raw.read_text().splitlines()] if raw.exists() else []
    done = {(r['suite'], r['seed'], r['seat'], r['policy']) for r in rows}
    if len(done) != len(rows) or not done <= expected: raise ValueError('Foreign or duplicate resumed score-curve game')
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
    write(out / 'status.json', {'status': 'loading_shared_ordinary_and_parent', 'games': len(rows)})
    ordinary = load(source['reference']); strong = Agent(ordinary)
    parent = load(source['warm_start']); parent_full = FullStage2Client(parent)
    parent_router = EncoderCallRouter(ordinary, parent_full)
    write(out / 'parent-native-parity.json', parity(ordinary, parent_full, parent_router,
        {'source_config': source['parent_config'], 'step': source['parent_step']}))
    started = time.monotonic()
    def play_policy(name, agent):
        with raw.open('a') as stream:
            for seed in seeds:
                for suite in SUITES:
                    for seat in range(4):
                        if (suite, seed, seat, name) in done: continue
                        row = {'policy': name, **game(agent, strong, seed, seat, suite)}
                        stream.write(json.dumps(row) + '\n'); stream.flush(); rows.append(row)
                        status = {'status': 'fresh_checkpoint_score_games', 'stage': cfg['stage'],
                            'policy': name, 'games': len(rows), 'total_games': len(expected),
                            'seconds': time.monotonic() - started}
                        write(out / 'status.json', status)
                        if len(rows) % 24 == 0: print(json.dumps(status), flush=True)
    play_policy('parent', Agent(parent_router))
    # At most oldB + parent + one continued model are resident at once.
    for step in steps:
        client = load(available[step]['checkpoint']); full = FullStage2Client(client)
        router = EncoderCallRouter(ordinary, full); agent = Agent(router)
        write(out / f'native-parity-{step:06d}.json', parity(ordinary, full, router,
            {'source_config': cfg['source_config'], 'step': step}))
        play_policy(policy_name(step), agent)
        del agent, router, full, client; gc.collect(); torch.cuda.empty_cache()
    verify_protocol(protocol)
    summary = summarize(rows, cfg, steps)
    write(out / 'summary.json', summary)
    write(out / 'status.json', {'status': 'fresh_score_games_complete_audit_pending', 'games': len(rows)})


if __name__ == '__main__': main()
