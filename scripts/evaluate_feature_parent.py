"""Compare an updated candidate with its exact parent on the same arena walls."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from clean_policy_evaluation import aggregate, SUITES
from feature_continuation import verify_preparation
from head_rl import write
from scripts.audit_guide_calls import compare


def registration(cfg):
    development = ROOT / cfg['out']
    audit = json.loads((development / 'evaluation-audit.json').read_text())
    dev = json.loads((development / 'protocol.json').read_text())
    source = json.loads((ROOT / cfg['source_config']).read_text())
    run = ROOT / source['out']
    preparation = json.loads((run / 'preparation.json').read_text())
    verify_preparation(ROOT, source, preparation)
    if dev['config'] != cfg or dev['training_config'] != source or audit['status'] != 'passed_complete':
        raise ValueError('Completed unchanged candidate development required')
    if digest(development / 'games-private.jsonl') != audit['raw_games_sha256']:
        raise ValueError('Candidate development games changed')
    if digest(run / 'result.json') != dev['training_result_sha256']:
        raise ValueError('Candidate selection changed')
    for name, expected in dev['candidate_files'].items():
        if digest(ROOT / dev['candidate_checkpoint'] / name) != expected:
            raise ValueError('Candidate weights changed')
    for name, expected in dev['code_sha256'].items():
        if digest(ROOT / name) != expected:
            raise ValueError('Candidate evaluation source changed')
    parent_cfg = json.loads((ROOT / source['parent_config']).read_text())
    parent = ROOT / parent_cfg['out']
    selected = json.loads((parent / 'result.json').read_text())
    if selected['checkpoint'] != source['warm_start'] or selected['selected_step'] != source['parent_step']:
        raise ValueError('Parent selection changed')
    files = {str((development / n).relative_to(ROOT)): digest(development / n)
             for n in ('protocol.json', 'evaluation-audit.json', 'games-private.jsonl')}
    files.update({str((run / n).relative_to(ROOT)): digest(run / n)
                  for n in ('protocol.json', 'result.json', 'preparation.json')})
    files[cfg['source_config']] = digest(ROOT / cfg['source_config'])
    code = dict(dev['code_sha256'])
    code['scripts/evaluate_feature_parent.py'] = digest(ROOT / 'scripts/evaluate_feature_parent.py')
    protocol = {'config': cfg, 'source_files': files, 'code_sha256': code,
        'parent_checkpoint': source['warm_start'], 'parent_files': preparation['warm_start_files'],
        'reference_files': preparation['reference_files'], 'reference_checkpoint': source['reference'],
        'candidate_checkpoint': dev['candidate_checkpoint'], 'candidate_files': dev['candidate_files'],
        'comparison': 'continued candidate minus frozen feature parent; candidate games reused exactly',
        'opponents': 'Same originalB/CallingOpponent/EfficiencyAgent seats as candidate development',
        'policy_scope': 'Both policies use ordinaryB; call decisions differ',
        'test_opened': False, 'deployed': False}
    return protocol, source, parent_cfg


def combined_rows(cfg):
    development = ROOT / cfg['out']
    candidates = [json.loads(line) for line in (development / 'games-private.jsonl').read_text().splitlines()]
    candidates = [r for r in candidates if r['policy'] == 'candidate']
    parents = [json.loads(line) for line in (ROOT / cfg['paired_parent_out'] / 'games-private.jsonl').read_text().splitlines()]
    if any(r['policy'] != 'reference' for r in parents):
        raise ValueError('Foreign policy in parent game log')
    return candidates + parents


def audit_comparison(cfg):
    out = ROOT / cfg['paired_parent_out']
    expected, _, _ = registration(cfg)
    if json.loads((out / 'protocol.json').read_text()) != expected:
        raise ValueError('Parent comparison protocol changed')
    rows = combined_rows(cfg)
    actual = aggregate(rows, cfg)
    compare(json.loads((out / 'summary.json').read_text()), actual)
    parity = json.loads((out / 'native-parity.json').read_text())
    ordinary = json.loads((out / 'ordinary-parity.json').read_text())
    if parity['checked'] != 6 or not parity['decisions_equal'] or parity['max_absolute_probability_error'] > cfg['parity_atol']:
        raise ValueError('Parent serving parity incomplete')
    if not ordinary['unchanged'] or ordinary['checks'] != 6:
        raise ValueError('Parent ordinary route changed')
    result = {'status': 'passed_complete', 'games': len(rows),
        'new_parent_games': len(rows) // 2, 'reused_candidate_games': len(rows) // 2,
        'overall_parent_paired_gain': actual['overall']['candidate_minus_reference'],
        'raw_parent_games_sha256': digest(out / 'games-private.jsonl'),
        'candidate_games_source_verified': True, 'source_identity_passed': True,
        'native_parity_passed': True, 'summary_recomputed': True,
        'human_test_opened': False, 'deployed': False}
    write(out / 'evaluation-audit.json', result)
    write(out / 'status.json', result)
    print(json.dumps(result), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    if args.audit_only:
        audit_comparison(cfg)
        return
    audit = json.loads((ROOT / cfg['out'] / 'evaluation-audit.json').read_text())
    if audit['status'] != 'passed_complete':
        raise ValueError('Full candidate development audit required')
    if audit['overall_paired_gain']['gain'] <= 0:
        print('Nonpositive development signal: additional parent games skipped.', flush=True)
        return
    protocol, source, parent_cfg = registration(cfg)
    out = ROOT / cfg['paired_parent_out']
    out.mkdir(parents=True, exist_ok=args.resume)
    target = out / 'protocol.json'
    if target.exists() and json.loads(target.read_text()) != protocol:
        raise ValueError('Resumed parent comparison differs')
    if not target.exists():
        write(target, protocol)
    import torch
    from jevany import JevModel
    from jevany.checkpoint import LoadOptions
    from jevany.inference import InferenceOptions
    from jevany.data import api_request
    from clean_policy_runtime import CleanPolicyRouter
    from feature_policy_runtime import FeatureClient
    from encoder_call_policy import EncoderCallRouter
    from scripts.evaluate_feature_policy import native_parity
    from scripts.evaluate_call_repair import Agent
    from scripts.evaluate_guide_calls import game
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    base = Path(os.environ.get('JEV_BASE_MODEL_DIR', Path.home() / '.cache/huggingface/hub/models--Qwen--Qwen3.5-0.8B/snapshots/2fc06364715b967f1860aea9cf38778875588b17'))
    def load(path):
        return JevModel.from_pretrained(str(ROOT / path), device='cuda', dtype='bf16',
            options=LoadOptions(merge=False, temperature=1., compile_mode=None, cuda_graphs=False,
                                fused_kernels=False, base_load_path=str(base)),
            inference_options=InferenceOptions(max_state_tokens=4096, max_branch_tokens=8192,
                                               max_packed_tokens=8192, prefix_cache_size=0))
    ordinary, parent_model = load(source['reference']), load(source['warm_start'])
    router = CleanPolicyRouter(ordinary, FeatureClient(parent_model))
    parent_run = ROOT / parent_cfg['out']
    preparation = json.loads((parent_run / 'preparation.json').read_text())
    parity = native_parity(router, parent_run, preparation, source['parent_step'],
        cfg['parity_per_phase_view'], cfg['parity_atol'], parent_cfg['views'])
    write(out / 'native-parity.json', parity)
    hybrid = EncoderCallRouter(ordinary, router)
    checks = 0
    with (ROOT / source['data'] / 'train.jsonl').open() as stream:
        for line in stream:
            record = json.loads(line)
            if record['_meta']['phase'] == 'reaction':
                continue
            request = api_request(record)
            before = ordinary.system_one(**request)['answers']['action']['probabilities']
            after = hybrid.system_one(**request)['answers']['action']['probabilities']
            if before != after:
                raise ValueError('Parent ordinary route differs')
            checks += 1
            if checks == 6:
                break
    if checks != 6:
        raise ValueError('Missing parent ordinary checks')
    write(out / 'ordinary-parity.json', {'checks': checks, 'unchanged': True})
    agent, strong = Agent(hybrid), Agent(ordinary)
    path = out / 'games-private.jsonl'
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    done = {(r['suite'], r['seed'], r['seat']) for r in rows}
    expected = {(s, n, p) for s in SUITES for n in range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds']) for p in range(4)}
    if len(done) != len(rows) or not done <= expected or any(r['policy'] != 'reference' for r in rows):
        raise ValueError('Duplicate or foreign resumed parent game')
    started = time.monotonic()
    with path.open('a') as stream:
        for suite in SUITES:
            for seed in range(cfg['seed_start'], cfg['seed_start'] + cfg['seeds']):
                for seat in range(4):
                    if (suite, seed, seat) in done:
                        continue
                    row = {'policy': 'reference', **game(agent, strong, seed, seat, suite)}
                    stream.write(json.dumps(row) + '\n')
                    stream.flush()
                    rows.append(row)
                    status = {'status': 'paired_parent_strength', 'games': len(rows),
                        'total_games': len(expected), 'seconds': time.monotonic() - started}
                    write(out / 'status.json', status)
                    print(json.dumps(status), flush=True)
    write(out / 'summary.json', aggregate(combined_rows(cfg), cfg))
    write(out / 'status.json', {'status': 'games_complete_audit_pending', 'games': len(rows)})


if __name__ == '__main__':
    main()
