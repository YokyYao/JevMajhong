"""Actual Mortal choices on independently reset native yaku source cases.

The teacher sees public observation/history only. Intended route is sampling
metadata, never an instruction or a presumed optimal action.
"""
import argparse
from collections import Counter, defaultdict
import copy
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha


def collect(config):
    gpu_only(); c = read(config)
    if (c['method'] != 'actual_mortal_public_initial_yaku_choices_v1'
            or c['teacher_root'] != '../jev-baseline-evaluation'
            or c['expected_source_cases'] != 512 or c['automatic_fit']
            or c['target_route_is_teacher_input']):
        raise ValueError('Exact public-only teacher collection recipe required')
    source = ROOT/c['source_summary']; summary = read(source)
    if (summary['status'] != 'actual_native_initial_yaku_scenarios_generated_unlabelled_not_fit_ready'
            or summary['planned_decisions'] != 512):
        raise ValueError('Complete actual unlabeled native pilot required')
    reg_path = ROOT/c['source_registration']; reg = read(reg_path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(2)
    from riichienv import RiichiEnv, GameRule, ActionType
    from jevmahjong.core.state import build_request, signature
    from tools.curriculum_mortal_source import MortalSource
    actor = MortalSource((ROOT/c['teacher_root']).resolve())
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    files = {str(p.resolve()): sha(p) for p in (Path(__file__), config, source, reg_path,
        source.parent/'registered-plans-private.json', source.parent/'completion-binding-private.json',
        ROOT/'cluster/curriculum-yaku-teacher-v1.slurm',
        ROOT/'docs/experiments/plans/curriculum-yaku-teacher-20261123.md')}
    files.update(actor.bindings)
    for split, spec in summary['files'].items():
        path = ROOT/spec['path']; files[str(path.resolve())] = sha(path)
        if files[str(path.resolve())] != spec['sha256']:
            raise ValueError('Unlabeled scenario source changed')
    write(out/'registered-private.json', dict(method=c['method'], config=c, files=files,
        source_partitions_unchanged=True, source_cases=512,
        teacher_weight_identity=actor.bindings,
        no_target_route_hints_or_true_wall_passed_to_teacher=True,
        job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname()))
    streams = {}; outputs = {}; coverage = {}; total = Counter(); grouped = {}; source_grouped = {}
    evidence = (out/'executed-first-actions-private.jsonl').open('x')
    forced = (out/'forced-rule-wins-private.jsonl').open('x')
    try:
        for split in ('train', 'validation'):
            outputs[split] = out/f'{split}-teacher-private.jsonl'
            streams[split] = outputs[split].open('x')
            counts = Counter(); routes = defaultdict(Counter); groups = set(); source_groups = set()
            path = ROOT/summary['files'][split]['path']
            for line in path.open():
                case = json.loads(line); original = case['source']; plan = original['plan']
                if plan['split'] != split:
                    raise ValueError('Original source family changed partition')
                env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=original['native_seed'])
                views = env.reset(oya=0, wall=original['wall'])
                request, menu, _ = build_request(views[0])
                if request != case['request'] or env.mjai_log != original['native_prefix']:
                    raise ValueError('Actual native public request/prefix did not reproduce source')
                # A fresh Bot/PlayerState per genuinely reset native case. No
                # original-history Bot is reused after changing concealed tiles.
                player = actor.player(0)
                chosen = player.act(views[0])
                label = next(k for k, action in zip(request['questions']['action']['criteria'], menu)
                             if signature(action) == signature(chosen))
                selected = request['questions']['action']['criteria'][label]['type']
                source_groups.add(plan['group_id']); counts['source_cases'] += 1
                routes[plan['route']]['source_cases'] += 1
                if chosen.action_type in (ActionType.RON, ActionType.TSUMO):
                    forced.write(json.dumps(dict(id=plan['id'], group_id=plan['group_id'], split=split,
                        source_route=plan['route'], selected=selected,
                        reason='existing_source_actor_forces_legal_win_without_network_choice'))+'\n')
                    counts['forced_rule_wins_excluded_from_teacher_SFT'] += 1
                    continue
                labelled = copy.deepcopy(request)
                labelled['questions']['action'].update(label=label,
                    src='curriculum3B:teacher:initial:'+plan['route'])
                labelled['_meta'] = dict(id=plan['id'], group_id=plan['group_id'], split=split,
                    phase='closed_discard', label_type='expert_replay', source='mortal_native_targeted_initial_choice',
                    scenario_route=plan['route'], scenario_removed_target_tiles=plan['removed_target_tiles'],
                    scenario_source=str(path.relative_to(ROOT)), scenario_source_id=plan['id'],
                    teacher_choice_not_optimality_proof=True, source_is_first_turn_dealer_only=True,
                    source_world_variant=plan['world_variant'])
                before = len(env.mjai_log)
                env.step({0: chosen})
                evidence.write(json.dumps(dict(id=plan['id'], split=split, label=label,
                    selected_action=json.loads(chosen.to_mjai()), native_suffix=env.mjai_log[before:],
                    native_action_executed=True))+'\n')
                streams[split].write(json.dumps(labelled, separators=(',', ':'))+'\n')
                groups.add(plan['group_id']); counts['teacher_labels'] += 1
                counts['selected:'+selected] += 1
                routes[plan['route']]['teacher_labels'] += 1
                routes[plan['route']]['selected:'+selected] += 1
                total['actual_teacher_labels'] += 1
                if total['actual_teacher_labels'] % 64 == 0:
                    print(json.dumps(dict(actual_teacher_labels=total['actual_teacher_labels'])), flush=True)
            if counts['source_cases'] != summary['files'][split]['records']:
                raise ValueError('Not all original partition cases were handled')
            streams[split].flush(); evidence.flush(); forced.flush()
            grouped[split] = groups; source_grouped[split] = source_groups
            coverage[split] = dict(counts=dict(counts), routes={k: dict(v) for k, v in routes.items()},
                groups=len(groups), source_groups=len(source_groups))
        if (sum(v['counts']['source_cases'] for v in coverage.values()) != 512
                or source_grouped['train'] & source_grouped['validation']):
            raise ValueError('Full512 source scope and whole-family holdouts required')
        if any(sha(p) != h for p, h in files.items()):
            raise ValueError('Bound source/teacher changed during collection')
        result = dict(status='actual_mortal_yaku_initial_choices_collected_not_stage3B_fit_admitted',
            method=c['method'], coverage=coverage, teacher_counts=dict(actor.counts),
            files={s: dict(path=str(p.relative_to(ROOT)), sha256=sha(p),
                records=coverage[s]['counts']['teacher_labels'], groups=len(grouped[s])) for s, p in outputs.items()},
            all512_public_requests_and_prefixes_reproduced=True,
            source_family_partitions_preserved=True, no_target_route_or_hidden_world_teacher_input=True,
            labels_are_actual_teacher_actions_not_route_optimality=True,
            confidence_not_measured=True, native_first_actions_executed=True,
            original_legal_forced_wins_separate_not_network_labels=True,
            midhand_threat_abandonment_and_complete_game_coverage_pending=True,
            data_normalization_tokenizer_and_3B_fit_recipe_pending=True,
            active_stage3A_unchanged=True, student_updates=0,
            raw_labels_source_walls_and_action_evidence_private=True,
            job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
        write(out/'summary-private.json', result)
        write(out/'completion-binding-private.json', dict(files=files))
        print(json.dumps(dict(status=result['status'], labels=total['actual_teacher_labels'])), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(type=type(error).__name__, error=str(error), counts=dict(total)))
        raise
    finally:
        evidence.close(); forced.close()
        for stream in streams.values(): stream.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(); collect(args.config.resolve())
