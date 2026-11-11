"""Reuse audited historical oldB/hybrid games; run only the missing full-stage2 arm."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from clean_policy_evaluation import aggregate as original_aggregate
from checkpoint_identity import completed_checkpoint_files
from head_rl import write
from scripts import evaluate_stage2_ablation as evaluation


def verified_controls(cfg):
    source = ROOT / cfg['reuse_from']
    protocol = json.loads((source / 'protocol.json').read_text())
    audit = json.loads((source / 'evaluation-audit.json').read_text())
    raw = source / 'games-private.jsonl'
    if (audit['status'] != 'passed_complete' or not audit['candidate_frozen']
            or not audit['source_identity_passed'] or not audit['native_parity_passed']
            or digest(raw) != audit['raw_games_sha256']):
        raise ValueError('Source historical games incomplete, changed, or unaudited')
    if protocol['candidate_files'] != completed_checkpoint_files(ROOT / cfg['checkpoint'], cfg['step']):
        raise ValueError('Historical hybrid used different stage2 weights')
    reference = json.loads((ROOT / cfg['reference_protocol']).read_text())['reference_files']
    if protocol['reference_files'] != reference or protocol['seed_scheme'] != 'spaced-v1':
        raise ValueError('Historical reference or wall scheme differs')
    for name, expected in protocol['code_sha256'].items():
        if digest(ROOT / name) != expected: raise ValueError('Historical arena source changed: ' + name)
    source_cfg = protocol['config']
    if (cfg['development_seed_start'] != source_cfg['seed_start']
            or cfg['development_seeds'] != cfg['reuse_first_walls']
            or not 1 < cfg['reuse_first_walls'] <= source_cfg['seeds']):
        raise ValueError('Reuse must take the predeclared first N complete walls, never selected by score')
    rows = [json.loads(l) for l in raw.read_text().splitlines()]
    recomputed = original_aggregate(rows, source_cfg)
    saved = json.loads((source / 'summary.json').read_text())
    if recomputed != saved: raise ValueError('Historical full-grid summary differs from independent recomputation')
    seeds = set(range(cfg['development_seed_start'], cfg['development_seed_start'] + cfg['development_seeds']))
    selected = []
    for row in rows:
        if row['seed'] not in seeds: continue
        if row['policy'] not in ('reference', 'candidate'): raise ValueError('Foreign historical policy')
        selected.append(dict(row, policy='reference' if row['policy'] == 'reference' else 'previous'))
    expected = cfg['development_seeds'] * 4 * 3 * 2
    if len(selected) != expected: raise ValueError('Missing reusable control games')
    return selected, protocol, audit


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True); p.add_argument('--resume', action='store_true')
    args = p.parse_args(); cfg = json.loads(args.config.read_text()); out = ROOT / cfg['out']
    selected, source_protocol, source_audit = verified_controls(cfg)
    out.mkdir(parents=True, exist_ok=args.resume)
    target = out / 'games-private.jsonl'
    if target.exists():
        rows = [json.loads(l) for l in target.read_text().splitlines()]
        existing = [r for r in rows if r['policy'] in ('reference', 'previous')]
        if existing != selected: raise ValueError('Cached controls changed during resume')
    else:
        target.write_text(''.join(json.dumps(r) + '\n' for r in selected))
    audit = {'status': 'verified_reused_controls', 'historical_games_verified': source_audit['games'],
        'reused_games': len(selected), 'new_full_stage2_games_planned': len(selected) // 2,
        'source_raw_sha256': source_audit['raw_games_sha256'],
        'source_audit_sha256': digest(ROOT / cfg['reuse_from'] / 'evaluation-audit.json'),
        'source_candidate_files': source_protocol['candidate_files'],
        'same_wall_seat_opponents_rules_and_weights': True,
        'wall_selection': 'first N logical walls including every seat and opponent suite; no score filtering',
        'interpretation': 'historical-wall diagnostic ablation, not a new independent confirmation'}
    write(out / 'reused-controls-audit.json', audit)
    # Only registration is specialized. Inference, common guards, complete games,
    # grid validation and aggregation use the exact frozen three-arm evaluator.
    def reuse_registration(actual_cfg):
        if actual_cfg != cfg: raise ValueError('Cached ablation config differs')
        verified_controls(cfg)
    evaluation.check_disjoint = reuse_registration
    evaluation.CODE = list(evaluation.CODE) + [
        'scripts/evaluate_stage2_cached_ablation.py', 'clean_policy_evaluation.py',
        str(args.config.resolve().relative_to(ROOT)),
        *(str((ROOT / cfg['reuse_from'] / n).relative_to(ROOT))
          for n in ('protocol.json', 'evaluation-audit.json', 'summary.json', 'games-private.jsonl'))]
    sys.argv = [str(ROOT / 'scripts/evaluate_stage2_ablation.py'), '--config', str(args.config), '--resume']
    evaluation.main()


if __name__ == '__main__': main()
