"""Finalize a public-state curriculum after a fully audited failed continuation.

Keep the established parent-origin, exposure, split and reaction-suite checks.
Only this new run's preparation is finalized before any GPU job is submitted.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files
from continuation_confirmation_gate import confirmation_decision
from feature_continuation import remaining_indices, verify_preparation
from head_rl import write
from scripts.prepare_reaction_continuation import main as prepare_reaction
from targeted_reaction_sampling import choose_requests


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, type=Path)
    cfg = json.loads(parser.parse_args().config.read_text())
    failed_confirmation = json.loads((ROOT / cfg['failed_continuation_confirmation_config']).read_text())
    decision = confirmation_decision(ROOT, failed_confirmation)
    if (decision['decision'] != 'skip_nonpositive_parent'
            or failed_confirmation['development_out'] != cfg['failed_continuation_development_out']):
        raise ValueError('Verified complete nonpositive continuation decision required')
    previous_cfg = json.loads((ROOT / cfg['exclude_executed_config']).read_text())
    previous = ROOT / previous_cfg['out']
    previous_preparation = json.loads((previous / 'preparation.json').read_text())
    verify_preparation(ROOT, previous_cfg, previous_preparation)
    development = ROOT / cfg['failed_continuation_development_out']
    audit = json.loads((development / 'parent-comparison/evaluation-audit.json').read_text())
    if audit['status'] != 'passed_complete' or audit['overall_parent_paired_gain']['gain'] > 0:
        raise ValueError('Completed nonpositive parent comparison required for this adjustment')
    if digest(development / 'parent-comparison/games-private.jsonl') != audit['raw_parent_games_sha256']:
        raise ValueError('Failed continuation evidence changed')
    protocol = json.loads((development / 'protocol.json').read_text())
    result = json.loads((previous / 'result.json').read_text())
    if protocol['candidate_checkpoint'] != result['checkpoint']:
        raise ValueError('Failed candidate selection differs')
    for name, expected in protocol['candidate_files'].items():
        if digest(ROOT / result['checkpoint'] / name) != expected:
            raise ValueError('Failed candidate identity changed')
    if previous_cfg['warm_start'] != cfg['warm_start']:
        raise ValueError('Adjustment must return to the same preserved parent')
    import torch
    executed = previous / f"training-run/checkpoints/step-{previous_cfg['max_steps']:06d}"
    executed_files = completed_checkpoint_files(executed, previous_cfg['max_steps'])
    state = torch.load(executed / 'trainer_state.pt', map_location='cpu', weights_only=True)
    args = state['args']
    order = state['order'].tolist()
    remaining_indices(order, previous_preparation['records']['train'], epoch=state['epoch'],
        next_microbatch=state['next_microbatch'], records_seen=state['records_seen'],
        batch=args['batch'], world_size=state['world_size'], p_none_pair=args['p_none_pair'])
    if state['optimizer_step'] != previous_cfg['max_steps'] or state['records_seen'] != cfg['exclude_executed_requests']:
        raise ValueError('Actual executed exposure differs')
    used = set(order[:state['records_seen']])
    excluded_ids = set()
    source = ROOT / previous_cfg['data'] / 'train.jsonl'
    for index, line in enumerate(source.open()):
        if index in used:
            excluded_ids.add(json.loads(line)['_meta']['id'])
    if len(excluded_ids) != state['records_seen']:
        raise ValueError('Prior executed ID coverage incomplete')

    # Includes clean parent initialization, positive parent independent evidence,
    # exact parent-exposure removal and the real reaction benchmark contract.
    prepare_reaction()
    data, out = ROOT / cfg['data'], ROOT / cfg['out']
    try:
        prepared = json.loads((out / 'preparation.json').read_text())
        verify_preparation(ROOT, cfg, prepared)
        if digest(data / 'train.jsonl') != digest(source):
            raise ValueError('Expected the same parent-unseen source pool')
        records = [json.loads(line) for line in (data / 'train.jsonl').open()]
        selected, coverage = choose_requests(records, excluded_ids, cfg['public_stratum_quotas'], cfg['seed'])
        if len(selected) != cfg['max_steps'] * 16:
            raise ValueError('Curriculum must cover the complete registered effective-batch budget')
        path = data / 'train.jsonl'
        with path.open('w') as stream:
            for record in selected:
                stream.write(json.dumps(record, ensure_ascii=False) + '\n')
        labels, phases = Counter(), Counter()
        for record in selected:
            q = record['questions']['action']
            labels[q['criteria'][q['label']]['type']] += 1
            phases[record['_meta']['phase']] += 1
        validation = [json.loads(line) for line in (data / 'validation.jsonl').open()]
        groups = {r['_meta']['group_id'] for r in selected}
        if groups & {r['_meta']['group_id'] for r in validation}:
            raise ValueError('Whole-match validation overlap')
        prepared.update(records={'train': len(selected), 'validation': len(validation)},
            train_actions=dict(labels), train_phases=dict(phases), train_matches=len(groups),
            public_state_curriculum=coverage, excluded_executed_requests=state['records_seen'],
            overlap_with_excluded_executed_requests=0,
            scope='New IDs versus all three feature-stage updates; existing TRAIN matches, not globally unseen history. Public eligibility/threat quotas; labels never filter sampling.')
        prepared['files']['train'] = digest(path)
        frozen = [ROOT / 'scripts/prepare_targeted_reaction_continuation.py',
                  ROOT / 'targeted_reaction_sampling.py', ROOT / cfg['exclude_executed_config'], source,
                  ROOT / cfg['failed_continuation_confirmation_config'],
                  ROOT / 'continuation_confirmation_gate.py',
                  ROOT / 'scripts/check_continuation_confirmation.py',
                  ROOT / 'feature_continuation_confirmation.slurm',
                  executed / 'trainer_state.pt', *(executed / name for name in executed_files)]
        frozen += [previous / name for name in ('protocol.json', 'result.json', 'preparation.json', 'token-admission.json')]
        frozen += [development / name for name in ('protocol.json', 'evaluation-audit.json', 'summary.json', 'games-private.jsonl',
                   'parent-comparison/protocol.json', 'parent-comparison/evaluation-audit.json',
                   'parent-comparison/summary.json', 'parent-comparison/games-private.jsonl')]
        for p in frozen:
            prepared['frozen_files'][str(p.relative_to(ROOT))] = digest(p)
        write(data / 'provenance.json', prepared)
        write(out / 'preparation.json', prepared)
        verify_preparation(ROOT, cfg, prepared)
        print(json.dumps({'complete': True, 'records': prepared['records'],
                          'train_actions': dict(labels), 'public_state_curriculum': coverage}), flush=True)
    except Exception as exc:
        write(out / 'invalidation.json', {'reason': 'Targeted preparation incomplete',
              'error': str(exc), 'training_allowed': False, 'deployed': False})
        raise


if __name__ == '__main__':
    main()
