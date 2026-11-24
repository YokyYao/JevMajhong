"""Fixed Mortal public-history choices on the expanded daiminkan source cohort."""
from collections import Counter
import copy
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_transfer import read, sha
from tools.generate_curriculum_kan_sources_v1 import restore
from tools.label_curriculum_kan_sources_v1 import observer_events, PublicHistoryObservation

RUN = ROOT/'runs/curriculum-daiminkan-context-v1'


def collect():
    allocation = resource(); gpu_only()
    launch_path = RUN/'teacher-launch-binding-v1-private.json'
    queue_path = RUN/'teacher-queue-v1-private.json'
    launch = read(launch_path); queue = read(queue_path)
    if (launch['status'] != 'contextual_daiminkan_teacher_frozen_before_execution'
            or queue['binding_sha256'] != sha(launch_path)
            or str(queue['teacher_job']) != os.environ['SLURM_JOB_ID']
            or any(sha(p) != h for p, h in launch['files'].items())):
        raise ValueError('Frozen actual source/teacher recipe and queued GPU job required')
    source_path = RUN/'sources-v1/summary-private.json'; source = read(source_path)
    binding_path = source_path.parent/'completion-binding-private.json'
    binding = read(binding_path)['files']
    total_cases = sum(s['records'] for s in source['files'].values())
    if (source['status'] != 'native_daiminkan_context_sources_generated_unlabelled_not_fit_admitted'
            or source['planned'] != 576 or total_cases+source['quarantined'] != 576
            or source['labels'] or source['student_updates']
            or not source['all_generated_native_prefixes_inputs_and_menus_replayed']
            or any(sha(p) != h for p, h in binding.items())):
        raise ValueError('Completed contextual source generation must remain immutable')
    reg_path = ROOT/'runs/curriculum-progression-parent-retention-v1/fit-plan/registered.json'
    reg = read(reg_path)
    sys.path[:0] = [reg['source'], reg['training_config']['upstream_source']]
    import torch
    torch.set_num_threads(1)
    from riichienv import ActionType
    from jevmahjong.core.state import build_request, legal_candidates, signature
    from tools.curriculum_mortal_source import MortalSource, public_events
    actor = MortalSource((ROOT/'../jev-baseline-evaluation').resolve())
    files = {**binding, **launch['files'], **actor.bindings}
    for p in (launch_path, queue_path, source_path, binding_path, reg_path):
        files[str(p.resolve())] = sha(p)
    out = RUN/'teacher-v1'; out.mkdir(exist_ok=False)
    write(out/'registered-private.json', dict(method='actual_public_mortal_daiminkan_context_choices_v1',
        allocation=allocation, files=files, source_cases=total_cases, positive_quota=False,
        full_private_source_history_not_teacher_input=True, student_updates=0))
    paths = {s: out/f'{s}-teacher-private.jsonl' for s in ('train', 'validation')}
    streams = {s: p.open('x') for s, p in paths.items()}
    evidence = (out/'submitted-native-actions-private.jsonl').open('x')
    forced = (out/'forced-rule-wins-private.jsonl').open('x')
    coverage = {}; groups = {}; ids = set(); totals = Counter()
    roles = {0: 'upper', 3: 'opposite', 2: 'lower'}
    try:
        for split in paths:
            counts = Counter(); families = set()
            source_file = ROOT/source['files'][split]['path']
            if sha(source_file) != source['files'][split]['sha256']:
                raise ValueError('Actual source case file changed')
            for line in source_file.open():
                case = json.loads(line); plan = case['source']['plan']; seat = case['source']['observer']
                if plan['split'] != split or plan['id'] in ids: raise ValueError('Source partition/identity differs')
                ids.add(plan['id']); env, obs = restore(case)
                request, menu, _ = build_request(obs)
                kinds = {v['type'] for v in request['questions']['action']['criteria'].values()}
                events = public_events(observer_events(env.mjai_log, seat), seat)
                player = actor.player(seat)
                chosen = player.act(PublicHistoryObservation(obs, events))
                cans = player.mirror.last_cans
                for kind, field in (('ankan', 'can_ankan'), ('daiminkan', 'can_daiminkan'),
                        ('kakan', 'can_kakan'), ('pon', 'can_pon'), ('chi', 'can_chi')):
                    if bool(getattr(cans, field)) != (kind in kinds):
                        raise ValueError('Native/teacher public-history eligibility mismatch: '+kind)
                counts['source_cases'] += 1; totals['source_cases'] += 1
                key = plan['context']+':'+roles[plan['discarder']]
                counts[key+':sources'] += 1
                if chosen.action_type in (ActionType.RON, ActionType.TSUMO):
                    forced.write(json.dumps(dict(id=plan['id'], group_id=plan['group_id'], split=split,
                        context=key, reason='existing_actor_forces_legal_win_not_network_label'))+'\n')
                    counts[key+':forced_wins'] += 1; counts['forced_wins'] += 1
                    totals['forced_wins'] += 1
                    continue
                keys = list(request['questions']['action']['criteria'])
                matches = [k for k, a in zip(keys, menu) if signature(a) == signature(chosen)]
                if len(matches) != 1: raise ValueError('Teacher action must have a unique genuine native identity')
                label = matches[0]; selected = request['questions']['action']['criteria'][label]['type']
                labelled = copy.deepcopy(request)
                labelled['questions']['action'].update(label=label,
                    src='curriculum_kan:teacher:daiminkan:'+key)
                labelled['_meta'] = dict(id=plan['id'], group_id=plan['group_id'], split=split,
                    phase='call_window', label_type='expert_replay',
                    source='mortal_public_contextual_daiminkan', source_context=plan['context'],
                    source_discarder_role=roles[plan['discarder']],
                    original_pilot_family_anchor=plan['original_pilot_family_anchor'],
                    source_case_file=str(source_file.relative_to(ROOT)), source_case_id=plan['id'],
                    teacher_choice_not_optimality_proof=True, scripted_prefix_actions_not_labels=True)
                batch = {}
                for p, view in env.get_obs_py(list(env.active_players)).items():
                    if p == seat: batch[p] = chosen
                    else:
                        actions = legal_candidates(view)
                        wins = [a for a in actions if a.action_type in (ActionType.RON, ActionType.TSUMO)]
                        passes = [a for a in actions if a.action_type == ActionType.PASS]
                        if not wins and len(passes) != 1: raise ValueError('Explicit native response required')
                        batch[p] = wins[0] if wins else passes[0]
                before = len(env.mjai_log); env.step(batch)
                evidence.write(json.dumps(dict(id=plan['id'], split=split, label=label,
                    context=key, submitted_action=json.loads(chosen.to_mjai()),
                    native_suffix=env.mjai_log[before:], complete_legal_batch_submitted=True,
                    simultaneous_priority_may_override_intention=True))+'\n')
                streams[split].write(json.dumps(labelled, separators=(',', ':'))+'\n')
                families.add(plan['group_id']); counts['teacher_labels'] += 1
                counts[key+':selected:'+selected] += 1
                totals['teacher_labels'] += 1; totals['selected:'+selected] += 1
            streams[split].flush(); coverage[split] = dict(counts); groups[split] = families
            if counts['source_cases'] != source['files'][split]['records']:
                raise ValueError('Every generated source must receive network label or explicit forced-win exclusion')
        evidence.flush(); forced.flush()
        if len(ids) != total_cases or groups['train'] & groups['validation']:
            raise ValueError('Complete fixed cohort and original family separation required')
        if any(sha(p) != h for p, h in files.items()): raise ValueError('Immutable source/teacher dependency changed')
        result = dict(status='actual_contextual_daiminkan_teacher_choices_collected_not_fit_admitted',
            source_cases=total_cases, coverage=coverage, totals=dict(totals), teacher_counts=dict(actor.counts),
            files={s: dict(path=str(p.relative_to(ROOT)), sha256=sha(p),
                records=coverage[s]['teacher_labels'], families=len(groups[s])) for s, p in paths.items()},
            all_native_inputs_menus_and_teacher_eligibility_checked=True,
            teacher_saw_public_prefix_and_own_hand_only=True,
            forced_rule_wins_excluded_from_network_SFT=True, positive_quota_or_replacement=False,
            source_groups_and_partitions_preserved=True, optimal_action_advantages_not_measured=True,
            global_corpus_normalization_encoder_and_fit_admission_pending=True,
            student_updates=0, active_control_recipe_unchanged=True, raw_private=True,
            job_id=os.environ['SLURM_JOB_ID'])
        write(out/'summary-private.json', result)
        for p in (*paths.values(), out/'registered-private.json', out/'summary-private.json',
                out/'submitted-native-actions-private.jsonl', out/'forced-rule-wins-private.jsonl'):
            files[str(p.resolve())] = sha(p)
        write(out/'completion-binding-private.json', dict(files=files))
        print(json.dumps(result), flush=True)
    except BaseException as error:
        write(out/'error-private.json', dict(error=repr(error), totals=dict(totals)))
        raise
    finally:
        evidence.close(); forced.close()
        for stream in streams.values(): stream.close()


if __name__ == '__main__': collect()
