"""Fixed Mortal decisions on genuine public-history kan opportunities.

These hard labels are teacher choices, not simulated action advantages or
expert-optimal labels. Source-prefix actions never become supervision.
"""
import argparse
from collections import Counter
import copy
import json
import os
from pathlib import Path
import socket
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.generate_curriculum_kan_sources_v1 import restore


def observer_events(prefix, seat):
    events = []
    for original in prefix:
        event = copy.deepcopy(json.loads(original) if isinstance(original, str) else original)
        if event['type'] == 'start_kyoku':
            event['tehais'] = [hand if p == seat else ['?']*13 for p, hand in enumerate(event['tehais'])]
        if event['type'] == 'tsumo' and event['actor'] != seat:
            event['pai'] = '?'
        events.append(event)
    return events


class PublicHistoryObservation:
    """Delegate native action mapping while giving a fresh Bot the full prefix."""
    def __init__(self, native, events):
        self.native = native
        self.events = events

    def __getattr__(self, name):
        return getattr(self.native, name)


def collect(config):
    allocation = resource(); gpu_only(); c = read(config)
    if (c['method'] != 'actual_mortal_public_kan_source_choices_v1'
            or c['teacher_root'] != '../jev-baseline-evaluation'
            or c['expected_source_cases'] != 189 or c['automatic_fit']
            or c['source_motif_is_teacher_input']):
        raise ValueError('Exact fixed teacher-only kan source collection required')
    launch_path = ROOT/'runs/curriculum-kan-sources-v1/teacher-launch-binding-v1-private.json'
    queue_path = ROOT/'runs/curriculum-kan-sources-v1/teacher-queue-v1-private.json'
    launch = read(launch_path); queue = read(queue_path)
    if (launch['status'] != 'kan_teacher_source_and_recipe_frozen_before_execution'
            or queue['binding_sha256'] != sha(launch_path)
            or str(queue['teacher_job']) != os.environ['SLURM_JOB_ID']
            or any(sha(p) != h for p, h in launch['files'].items())):
        raise ValueError('Actual frozen source/recipe and submitted GPU job required')
    source_path = ROOT/c['source_summary']; source = read(source_path)
    binding_path = source_path.parent/'completion-binding-private.json'
    source_binding = read(binding_path)['files']
    if (source['status'] != 'native_kan_source_pilot_generated_unlabelled_not_fit_admitted'
            or source['planned'] != 192 or source['quarantined'] != 3
            or sum(spec['records'] for spec in source['files'].values()) != 189
            or source['labels'] or source['student_updates']
            or any(sha(p) != h for p, h in source_binding.items())):
        raise ValueError('Actual immutable189-case source pilot required')
    reg_path = ROOT/c['source_registration']; reg = read(reg_path)
    sys.path.insert(0, reg['source']); sys.path.insert(0, reg['training_config']['upstream_source'])
    import torch
    torch.set_num_threads(1)
    from riichienv import ActionType
    from jevmahjong.core.state import build_request, legal_candidates, signature
    from tools.curriculum_mortal_source import MortalSource, public_events
    actor = MortalSource((ROOT/c['teacher_root']).resolve())
    files = dict(source_binding)
    for p in (Path(__file__), config, source_path, binding_path, reg_path, launch_path, queue_path,
              ROOT/'cluster/curriculum-kan-teacher-v1.slurm',
              ROOT/'docs/experiments/plans/curriculum-kan-teacher-20261124.md'):
        files[str(p.resolve())] = sha(p)
    files.update(actor.bindings)
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=False)
    write(out/'registered-private.json', dict(method=c['method'], config=c, files=files,
        source_partitions_unchanged=True, source_cases=189, allocation=allocation,
        no_motif_true_wall_or_opponent_hands_passed_to_teacher=True,
        source_prefix_actions_not_supervision=True))
    streams = {}; outputs = {}; coverage = {}; groups = {}; identifiers = set(); total = Counter()
    evidence = (out/'submitted-native-actions-private.jsonl').open('x')
    forced = (out/'forced-rule-wins-private.jsonl').open('x')
    try:
        for split in ('train', 'validation'):
            outputs[split] = out/f'{split}-teacher-private.jsonl'
            streams[split] = outputs[split].open('x'); counts = Counter(); families = set()
            source_file = ROOT/source['files'][split]['path']
            if sha(source_file) != source['files'][split]['sha256']:
                raise ValueError('Source case file changed')
            for line in source_file.open():
                case = json.loads(line); plan = case['source']['plan']; seat = case['source']['observer']
                if plan['split'] != split or plan['id'] in identifiers:
                    raise ValueError('Source split/unique decision identity changed')
                identifiers.add(plan['id']); env, obs = restore(case)
                request, menu, _ = build_request(obs)
                kinds = {value['type'] for value in request['questions']['action']['criteria'].values()}
                events = public_events(observer_events(env.mjai_log, seat), seat)
                player = actor.player(seat)
                selected = player.act(PublicHistoryObservation(obs, events))
                cans = player.mirror.last_cans
                for kind, field in (('ankan', 'can_ankan'), ('daiminkan', 'can_daiminkan'),
                                    ('kakan', 'can_kakan'), ('pon', 'can_pon'), ('chi', 'can_chi')):
                    if bool(getattr(cans, field)) != (kind in kinds):
                        raise ValueError('Native/Mortal public-history legal eligibility differs: '+kind)
                counts['source_cases'] += 1; total['source_cases'] += 1
                counts['source_operation:'+plan['operation']] += 1
                if selected.action_type in (ActionType.RON, ActionType.TSUMO):
                    forced.write(json.dumps(dict(id=plan['id'], group_id=plan['group_id'], split=split,
                        source_operation=plan['operation'], reason='existing_source_actor_forces_legal_win'))+'\n')
                    counts['forced_rule_wins_excluded_from_SFT'] += 1
                    continue
                keys = list(request['questions']['action']['criteria'])
                matches = [key for key, action in zip(keys, menu) if signature(action) == signature(selected)]
                if len(matches) != 1:
                    raise ValueError('Teacher choice lacks a unique native legal-menu identity')
                label = matches[0]; selected_type = request['questions']['action']['criteria'][label]['type']
                labelled = copy.deepcopy(request)
                labelled['questions']['action'].update(label=label,
                    src='curriculum_kan:teacher:'+plan['operation']+':'+plan['motif'])
                labelled['_meta'] = dict(id=plan['id'], group_id=plan['group_id'], split=split,
                    phase='call_window' if plan['operation'] == 'daiminkan' else 'kan_window',
                    label_type='expert_replay', source='mortal_public_scripted_native_kan_opportunity',
                    source_operation=plan['operation'], source_motif=plan['motif'],
                    source_case_file=str(source_file.relative_to(ROOT)), source_case_id=plan['id'],
                    source_world_variant=plan['world_variant'], teacher_choice_not_optimality_proof=True,
                    scripted_prefix_actions_not_supervision=True)
                # Preserve actual submitted intentions, including other actors'
                # legal wins. A call can be superseded by simultaneous priority;
                # that does not erase the observed teacher's intention.
                batch = {}
                for p, view in env.get_obs_py(list(env.active_players)).items():
                    if p == seat:
                        batch[p] = selected
                    else:
                        actions = legal_candidates(view)
                        wins = [a for a in actions if a.action_type in (ActionType.RON, ActionType.TSUMO)]
                        passes = [a for a in actions if a.action_type == ActionType.PASS]
                        if not wins and len(passes) != 1:
                            raise ValueError('Other response actor needs an explicit legal win or pass')
                        batch[p] = wins[0] if wins else passes[0]
                before = len(env.mjai_log); env.step(batch)
                evidence.write(json.dumps(dict(id=plan['id'], split=split, label=label,
                    submitted_action=json.loads(selected.to_mjai()), native_suffix=env.mjai_log[before:],
                    complete_legal_batch_submitted=True, simultaneous_priority_may_override_intention=True))+'\n')
                streams[split].write(json.dumps(labelled, separators=(',', ':'))+'\n')
                families.add(plan['group_id']); counts['teacher_labels'] += 1
                counts[plan['operation']+':selected:'+selected_type] += 1
                counts['motif:'+plan['motif']+':selected:'+selected_type] += 1
                total['teacher_labels'] += 1
                if total['teacher_labels'] % 24 == 0:
                    streams[split].flush(); evidence.flush(); forced.flush()
                    print(json.dumps(dict(actual_teacher_labels=total['teacher_labels'],
                        processed=total['source_cases'], total=189)), flush=True)
            if counts['source_cases'] != source['files'][split]['records']:
                raise ValueError('Incomplete original source partition')
            streams[split].flush(); coverage[split] = dict(counts); groups[split] = families
        evidence.flush(); forced.flush()
        if len(identifiers) != 189 or groups['train'] & groups['validation']:
            raise ValueError('All source cases and whole-family split required')
        if any(sha(p) != h for p, h in files.items()):
            raise ValueError('A bound source or teacher dependency changed')
        result = dict(status='actual_mortal_kan_source_choices_collected_not_fit_admitted',
            coverage=coverage, teacher_counts=dict(actor.counts),
            files={s: dict(path=str(p.relative_to(ROOT)), sha256=sha(p),
                records=coverage[s]['teacher_labels'], families=len(groups[s])) for s, p in outputs.items()},
            source_cases=189, all189_native_public_inputs_and_menus_reproduced=True,
            teacher_kan_chi_pon_flags_match_native_for_all_sources=True,
            source_motif_and_hidden_world_not_teacher_inputs=True,
            source_prefix_actions_not_labels=True, forced_rule_wins_excluded_from_teacher_SFT=True,
            native_teacher_intentions_submitted=True, labels_are_teacher_choices_not_optimality=True,
            full_corpus_overlap_normalization_encoder_and_fit_admission_pending=True,
            conditional_public_world_action_advantage_not_measured=True,
            current_training_and_published_model_unchanged=True, student_updates=0, raw_private=True,
            job_id=os.environ['SLURM_JOB_ID'], node=socket.gethostname())
        write(out/'summary-private.json', result)
        for p in (out/'registered-private.json', out/'summary-private.json',
                  out/'submitted-native-actions-private.jsonl', out/'forced-rule-wins-private.jsonl', *outputs.values()):
            files[str(p.resolve())] = sha(p)
        write(out/'completion-binding-private.json', dict(files=files))
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(type=type(error).__name__, error=str(error), counts=dict(total)))
        raise
    finally:
        evidence.close(); forced.close()
        for stream in streams.values():
            stream.close()


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    collect(p.parse_args().config.resolve())
