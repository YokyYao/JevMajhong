"""Paired complete-East oldB / call-only stage2 / full stage2 comparison."""
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
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files
from encoder_call_policy import EncoderCallRouter
from guide_development_audit import aggregate, SUITES, POLICIES
from head_rl import write
from scripts.collect_feature_points import check_disjoint
from scripts.evaluate_call_repair import Agent
from scripts.evaluate_guide_calls import game
from stage2_ablation import FullStage2Client

CODE = ['stage2_ablation.py', 'scripts/evaluate_stage2_ablation.py',
    'scripts/audit_stage2_ablation.py', 'scripts/evaluate_guide_calls.py',
    'scripts/evaluate_call_repair.py', 'scripts/train_call_repair.py',
    'guide_development_audit.py', 'clean_policy_data.py', 'call_option_features.py',
    'guide_call_features.py', 'call_policy.py', 'encoder_call_policy.py', 'semantic.py',
    'arena.py', 'scoring_audit.py', 'seed_control.py', 'checkpoint_identity.py',
    'vendor/JevAny/jevany/inference.py', 'vendor/JevAny/jevany/data.py',
    'vendor/JevAny/jevany/model.py', 'vendor/JevAny/jevany/client.py']


def verify_protocol(protocol):
    cfg = protocol['config']
    for name, expected in protocol['code_sha256'].items():
        if digest(ROOT / name) != expected: raise ValueError('Ablation source changed: ' + name)
    for name, expected in protocol['reference_files'].items():
        if digest(ROOT / cfg['reference'] / name) != expected: raise ValueError('oldB changed')
    if completed_checkpoint_files(ROOT / cfg['checkpoint'], cfg['step']) != protocol['stage2_files']:
        raise ValueError('Stage2 changed')


def parity(ordinary, full, hybrid, cfg):
    from jevany.data import api_request
    source = json.loads((ROOT / cfg['source_config']).read_text())
    run, data = ROOT / source['out'], ROOT / source['data']
    prepared = json.loads((run / 'preparation.json').read_text())
    if digest(data / 'validation.jsonl') != prepared['files']['validation']:
        raise ValueError('Existing validation changed')
    saved = json.loads((run / 'training-run/training_eval' /
        f"step-{cfg['step']:06d}" / 'development/rows.json').read_text())
    expected = {r['id']: r for r in saved}
    counts = {'reaction': 0, 'open_discard': 0, 'closed_discard': 0}
    maximum = 0.; ordinary_checks = 0
    for line in (data / 'validation.jsonl').open():
        record = json.loads(line); meta = record['_meta']; phase = meta['phase']
        if meta['view'] != 'native' or counts[phase] >= 6: continue
        request = api_request(record); keys = list(request['questions']['action']['criteria'])
        target_row = expected[meta['id']]
        p = full.system_one(**request)['answers']['action']['probabilities']
        if list(p) != keys or keys != target_row['keys']: raise ValueError('Stage2 option order differs')
        actual, target = torch.tensor([p[k] for k in keys]), torch.tensor(target_row['p'])
        error = float((actual - target).abs().max()); maximum = max(maximum, error)
        if error > 2e-5 or actual.argmax() != target.argmax():
            raise ValueError('Full stage2 native probabilities differ from training validation')
        mixed = hybrid.system_one(**request)['answers']['action']['probabilities']
        if phase == 'reaction':
            if mixed != p: raise ValueError('Full and hybrid call behavior differs')
        else:
            baseline = ordinary.system_one(**request)['answers']['action']['probabilities']
            if mixed != baseline: raise ValueError('Hybrid ordinary behavior differs from oldB')
            ordinary_checks += 1
        counts[phase] += 1
    if set(counts.values()) != {6}: raise ValueError('Parity phase coverage incomplete')
    return {'native_counts': counts, 'maximum_probability_error': maximum,
        'full_stage2_decisions_equal_to_saved_validation': True,
        'hybrid_call_probabilities_equal_full_stage2': True,
        'hybrid_ordinary_probabilities_equal_oldB': ordinary_checks,
        'source': 'already-opened validation; no sealed human test'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--resume', action='store_true')
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    out.mkdir(parents=True, exist_ok=args.resume)
    lock = (out / 'evaluation.lock').open('w'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    source = json.loads((ROOT / cfg['reference_protocol']).read_text())
    protocol = {'config': cfg, 'stage2_files': completed_checkpoint_files(ROOT / cfg['checkpoint'], cfg['step']),
        'reference_files': source['reference_files'], 'code_sha256': {n: digest(ROOT / n) for n in CODE},
        'arms': {'reference': 'oldB all model-scored decisions',
            'previous': 'oldB ordinary/rare; stage2 chi/pon/pass (historically confirmed hybrid)',
            'candidate': 'stage2 all model-scored decisions; no oldB fallback'},
        'common_rules': 'same automatic wins and ordinary discard guard in Agent; same legal candidates',
        'strong_opponent': 'frozen oldB at focal+1; same in every arm',
        'candidate_frozen': True, 'human_test_opened': False,
        'primary': 'full stage2 minus hybrid complete-East net points, whole-wall paired CI',
        'deployed': False}
    target = out / 'protocol.json'
    if target.exists():
        if json.loads(target.read_text()) != protocol: raise ValueError('Resume protocol differs')
    else:
        check_disjoint(cfg); write(target, protocol)
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
    write(out / 'status.json', {'status': 'loading_frozen_stage2_ablation', 'games': 0})
    ordinary, candidate = load(cfg['reference']), load(cfg['checkpoint'])
    full = FullStage2Client(candidate); hybrid = EncoderCallRouter(ordinary, full)
    write(out / 'native-parity.json', parity(ordinary, full, hybrid, cfg))
    agents = {'reference': Agent(ordinary), 'previous': Agent(hybrid), 'candidate': Agent(full)}
    strong = Agent(ordinary); raw = out / 'games-private.jsonl'
    rows = [json.loads(l) for l in raw.read_text().splitlines()] if raw.exists() else []
    done = {(r['suite'], r['seed'], r['seat'], r['policy']) for r in rows}
    seeds = range(cfg['development_seed_start'], cfg['development_seed_start'] + cfg['development_seeds'])
    expected = {(s, n, p, k) for s in SUITES for n in seeds for p in range(4) for k in POLICIES}
    if len(done) != len(rows) or not done <= expected: raise ValueError('Duplicate or foreign resumed games')
    started = time.monotonic()
    with raw.open('a') as stream:
        for seed in seeds:
            for suite in SUITES:
                for seat in range(4):
                    for name in POLICIES:
                        if (suite, seed, seat, name) in done: continue
                        row = {'policy': name, **game(agents[name], strong, seed, seat, suite)}
                        stream.write(json.dumps(row) + '\n'); stream.flush(); rows.append(row)
                        status = {'status': 'stage2_full_vs_hybrid_games', 'games': len(rows),
                            'total_games': len(expected), 'suite': suite, 'seconds': time.monotonic() - started}
                        write(out / 'status.json', status)
                        if len(rows) % 36 == 0: print(json.dumps(status), flush=True)
    verify_protocol(protocol)
    summary = aggregate(rows, cfg)
    summary.update(primary='full stage2 minus preserved hybrid; whole-wall paired terminal net points',
        policy_names=protocol['arms'])
    summary.pop('development_positive_signal', None); summary.pop('meets_development_stability_condition', None)
    write(out / 'summary.json', summary)
    write(out / 'status.json', {'status': 'ablation_games_complete_audit_pending', 'games': len(rows)})


if __name__ == '__main__': main()
