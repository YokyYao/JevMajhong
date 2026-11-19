"""Collect isolated native East teacher menus inside an allocated GPU job."""
import argparse
from collections import Counter
import copy
import hashlib
import json
import os
from pathlib import Path
import socket
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def verify(proof):
    for path, expected in proof['files'].items():
        if sha(path) != expected:
            raise ValueError('Registered artifact changed: ' + path)


def registration(c, extra=()):
    # Hash transitive project modules, not only the launch script. All hashing
    # is in the allocated GPU job's companion CPUs.
    files = [ROOT/'configs/training/mortal-teacher-v1.json',
             ROOT/'docs/experiments/plans/mortal-teacher-20261119.md',
             ROOT/'tools/build_mortal_teacher.py', ROOT/'tools/check_gpu.py']
    files += list((ROOT/'jevmahjong').rglob('*.py'))
    files += list((ROOT/'desktop/jev_sidecar').glob('*.py'))
    files += [ROOT/p for p in extra]
    teacher = Path(c['teacher_root'])
    files += [teacher/'mortal-upstream/mortal/model.py', teacher/'mortal-upstream/mortal/engine.py',
              teacher/'mortal-adapter-reference/_libriichi_loader.py']
    return dict(files={str(p): sha(p) for p in sorted(set(files))}, config=copy.deepcopy(c))


def bucket(criteria, selected, threats):
    kinds = {v['type'] for v in criteria.values()}
    if 'hora' in kinds:
        return None
    chosen = criteria[selected]['type']
    if kinds & {'ankan', 'daiminkan', 'kakan'}:
        return 'teacher_kan'
    if chosen in {'reach', 'chi', 'pon'}:
        return 'teacher_' + chosen
    if chosen == 'none' and kinds & {'chi', 'pon'}:
        return 'teacher_reaction'
    if chosen == 'dahai':
        return 'teacher_defense' if threats else 'teacher_discard'
    return None


def smoothed_target(keys, selected, epsilon):
    if selected not in keys or len(set(keys)) != len(keys) or not 0 <= epsilon < 1:
        raise ValueError('Invalid teacher label or smoothing')
    return {k: epsilon/len(keys) + (1-epsilon if k == selected else 0.) for k in keys}


def select_rows(rows, quotas, cap):
    """Rare categories first, ordered only by public-source identity hash."""
    chosen = []; per_group = Counter(); ids = set()
    for source in ('teacher_kan', 'teacher_chi', 'teacher_pon', 'teacher_reach',
                   'teacher_defense', 'teacher_reaction', 'teacher_discard'):
        pool = sorted((r for r in rows if r['questions']['action']['src'] == source),
                      key=lambda r: r['_meta']['selection_hash'])
        got = 0
        for r in pool:
            if got == quotas[source]:
                break
            rid, group = r['_meta']['id'], r['_meta']['group_id']
            if rid in ids or per_group[group] >= cap:
                continue
            chosen.append(r); ids.add(rid); per_group[group] += 1; got += 1
        if got != quotas[source]:
            raise ValueError(f'Teacher coverage shortfall for {source}: {got}/{quotas[source]}')
    return chosen


def assert_split(train, validation):
    if ({r['_meta']['group_id'] for r in train} & {r['_meta']['group_id'] for r in validation}
            or {r['_meta']['id'] for r in train} & {r['_meta']['id'] for r in validation}):
        raise ValueError('Teacher source game crosses train/validation')


def require_gpu():
    if (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS')
            or socket.gethostname().startswith(('lo-', 'login'))):
        raise RuntimeError('Computation requires an allocated GPU node')
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Allocated CUDA device is unavailable')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def source_match(c, teacher, seed, split):
    from riichienv import RiichiEnv, GameRule
    from jevmahjong.core.seeds import engine_seed
    from jevmahjong.core.state import build_request, legal_candidates
    from jevmahjong.core.scoring import settlement_ledger
    from jevmahjong.evaluation.agents import EfficiencyAgent, CallingOpponent
    from desktop.jev_sidecar.unified_policy import serving_request
    from jevmahjong.endgame.public_policy import original_policy_keys
    from jevmahjong.endgame.collect import digest
    from jevmahjong.teachers.mortal import legal_index, WEIGHT_SHA256
    env = RiichiEnv(game_mode='4p-red-east', rule=GameRule.default_mjsoul(), seed=engine_seed(seed, 'spaced-v1'))
    observations = env.reset(); seat = seed % 4; profile = seed % 3
    players = {p: teacher.player(p) for p in range(4)}
    rules = [EfficiencyAgent(), CallingOpponent()]
    pool = []; rejected = Counter(); decisions = Counter(); windows = Counter()
    for step in range(30000):
        if env.done():
            break
        chosen = {}
        for p, obs in observations.items():
            relative = (p-seat) % 4
            mortal = p == seat or profile == 0 or (profile == 1 and relative < 3) or (profile == 2 and relative == 1)
            action = players[p].act(obs) if mortal else rules[relative % 2].act(obs)
            chosen[p] = action
            if p != seat:
                continue
            decisions[json.loads(action.to_mjai())['type']] += 1
            actions = legal_candidates(obs)
            if len(actions) < 2:
                continue
            if int(obs.round_wind) != 0 or int(obs.kyoku_index) >= 3:
                rejected['outside_East1_3'] += 1; continue
            request, actions, _ = build_request(obs)
            request['state']['hand_number'] = int(obs.kyoku_index)+1
            payload = serving_request(request)
            keys = list(payload['questions']['action']['criteria'])
            index = legal_index(action, actions); selected = keys[index]
            q = payload['questions']['action']
            kinds = {v['type'] for v in q['criteria'].values()}
            for kind in kinds & {'ankan','daiminkan','kakan'}:
                windows[kind] += 1
            if 'hora' in kinds:
                rejected['win_menu_retains_East_objective'] += 1; continue
            if selected not in original_policy_keys(payload):
                rejected['outside_unchanged_serving_guard'] += 1; continue
            threats = [p for p in payload['state']['riichi'] if p != seat]
            category = bucket(q['criteria'], selected, threats)
            if category is None:
                rejected['unregistered_action_class'] += 1; continue
            q['src'] = category
            q['target'] = smoothed_target(keys, selected, c['teacher_label_smoothing'])
            payload['_meta'] = dict(id=f'mortal-teacher:{seed}:{step}:{seat}', group_id=f'mortal-source:{seed}',
                seed=seed, step=step, seat=seat, split=split, teacher_action=selected,
                teacher_checkpoint_sha256=WEIGHT_SHA256, profile=profile,
                kan_types=sorted(kinds & {'ankan','daiminkan','kakan'}),
                label_type='mapped_teacher_action_not_terminal_probability',
                selection_hash=digest([seed, step, seat, category, 'mortal-teacher-v1']))
            pool.append(payload)
        observations = env.step(chosen)
    if not env.done():
        raise RuntimeError('Teacher source match failed to terminate')
    ledger = settlement_ledger(env.mjai_log, env.scores())
    return dict(seed=seed, split=split, pool=pool, rejected=dict(rejected), decisions=dict(decisions),
        kan_windows=dict(windows), final_scores=list(env.scores()), final_ranks=list(env.ranks()),
        ledger=ledger, mjai_log=env.mjai_log, completed=True)


def collect(c, proof):
    from jevmahjong.endgame.collect import write
    from jevmahjong.endgame.prepare import read_jsonl
    from jevmahjong.teachers.mortal import MortalTeacher, WEIGHT_SHA256
    out = ROOT/c['out']; out.mkdir(parents=True, exist_ok=True)
    verify(proof)
    old = json.loads((ROOT/c['retention_root']/'data-ready-private.json').read_text())
    for name, checksum in old['data_sha256'].items():
        if sha(ROOT/c['retention_root']/(name+'.jsonl')) != checksum:
            raise ValueError('Retained endgame data changed')
    audit = json.loads((ROOT/c['retention_root']/'training-audit-private.json').read_text())
    if not any(x['total_step']==176 and x['eligible_selected_for_consumed_development'] for x in audit['audits']):
        raise ValueError('Parent176 is not the audited selected research checkpoint')
    write(out/'registered-data-code-private.json', proof)
    teacher = MortalTeacher(c['teacher_root']); pools = {}; stats = Counter(); started = time.monotonic()
    write(out/'teacher-proof-private.json', dict(weight_sha256=WEIGHT_SHA256, architecture='v4_resnet192x40',
        weights_only=True, strict_state_dict=True, oracle=False, agari_guard=False,
        training_rank_points=teacher.config['env']['pts'], node=socket.gethostname(),job_id=os.environ['SLURM_JOB_ID']))
    for split in ('train','validation'):
        rows = []
        for group in range(c[split+'_groups']):
            seed = c[split+'_seed_start']+group; path = out/'sources'/f'{seed}.json'
            if path.exists():
                source = json.loads(path.read_text())
                if source['seed'] != seed or source['split'] != split or not source['completed']:
                    raise ValueError('Existing source differs')
            else:
                source = source_match(c,teacher,seed,split); write(path,source)
            rows.extend(source['pool']); stats.update({split+':'+k:v for k,v in source['rejected'].items()})
            if group%8==0 or group+1==c[split+'_groups']:
                status=dict(status='collecting_teacher_native_games',split=split,groups_complete=group+1,
                    groups_total=c[split+'_groups'],positions=len(rows),rejections=dict(stats),
                    seconds=time.monotonic()-started,job_id=os.environ['SLURM_JOB_ID'])
                write(out/'collection-status-private.json',status); print(json.dumps(status),flush=True)
        pools[split] = rows
    selected={s:select_rows(pools[s],c[s+'_quotas'],c['source_group_cap']) for s in pools}
    assert_split(selected['train'],selected['validation'])
    coverage={s:Counter(k for r in selected[s] for k in r['_meta']['kan_types']) for s in selected}
    if any(coverage[s][k] < (4 if s=='train' else 1) for s in selected for k in ('ankan','daiminkan','kakan')):
        raise ValueError('Missing required genuine kan windows; do not fabricate teacher labels')
    oldtrain=read_jsonl(ROOT/c['retention_root']/'train.jsonl')
    if len(oldtrain)!=874:
        raise ValueError('Keep all874 existing positive training records')
    for r in oldtrain:
        prior=r['questions']['action']['src']
        r['questions']['action']['src']='retention_replay' if prior in ('old_replay','old_leading') else 'retention_endgame'
        r['_meta']['retention_original_src']=prior
    train=selected['train']+oldtrain; validation=selected['validation']
    for name, rows in [('train',train),('validation',validation)]:
        path=out/(name+'.jsonl')
        path.write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
    from jevmahjong.endgame.futures import bundle_identity
    result=dict(status='mortal_teacher_data_ready',train=len(train),teacher_train=len(selected['train']),
        teacher_validation=len(validation),retention=len(oldtrain),rejections=dict(stats),
        train_counts=dict(Counter(r['questions']['action']['src'] for r in train)),
        validation_counts=dict(Counter(r['questions']['action']['src'] for r in validation)),
        kan_windows={s:dict(coverage[s]) for s in coverage},
        kan_choices={s:dict(Counter(r['questions']['action']['criteria'][r['_meta']['teacher_action']]['type']
            for r in selected[s] if r['questions']['action']['src']=='teacher_kan')) for s in selected},
        teacher_model_sha256=WEIGHT_SHA256,parent_identity=bundle_identity(ROOT/c['parent_checkpoint']),
        files={str(out/(s+'.jsonl')):sha(out/(s+'.jsonl')) for s in selected},
        source_groups={s:len({r['_meta']['group_id'] for r in selected[s]}) for s in selected},
        whole_game_split=True,ledger_checks=True,raw_publication=False,release_qualified=False)
    # Training source has additional retention records; both hashes are recorded above.
    verify(proof); write(out/'data-ready-private.json',result); print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True);a=p.parse_args()
    require_gpu();c=json.loads(a.config.read_text());os.chdir(ROOT)
    collect(c,registration(c,('tests/core/test_mortal_teacher.py',)))
