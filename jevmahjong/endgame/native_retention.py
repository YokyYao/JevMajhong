"""Admit paired native contexts without inventing expert or terminal labels."""
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from .collect import digest
from .context import annotate_record,public_context,TRAILING_INPUT_VERSION
from .public_policy import original_policy_keys,action_keys
from desktop.jev_sidecar.unified_policy import serving_request


def load_native_pairs(config):
    if not config.get('native_context_root'):return []
    root=Path(config['native_context_root'])
    summary=json.loads((root/'summary-private.json').read_text())
    rows=json.loads((root/'paired-contexts-private.json').read_text())
    if (summary['status']!='native_ordinary_context_manifest_ready_no_teacher_queries'
            or summary['positions']!=80 or len(rows)!=80 or len({r['seed'] for r in rows})!=80
            or digest(rows)!=config['native_context_positions_sha256']
            or summary['positions_sha256']!=config['native_context_positions_sha256']):
        raise ValueError('Native context manifest is not the registered isolated80')
    for path,expected in summary['source_hashes'].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError('Original native context source changed')
    counts=Counter()
    for row in rows:
        split=row['split'];rank=row['rank'];counts[split+':'+str(rank)]+=1
        if row['group_id']!='endgame-native:'+str(row['seed']) or rank not in (3,4):
            raise ValueError('Native context source group or placement differs')
        if split!=('validation' if (row['seed']-2026116900)%4==3 else 'train'):
            raise ValueError('Native context original split differs')
        teacher=serving_request(row['raw_request']);student=row['student_request']
        if teacher!=row['teacher_request'] or digest(teacher)!=row['teacher_request_sha256'] or digest(student)!=row['student_request_sha256']:
            raise ValueError('Native teacher or student input changed')
        native=copy.deepcopy(row['raw_request'])
        screen=student['state'].get('screen_observation')
        if screen is not None:
            if not isinstance(screen,dict) or set(screen)-{'reaction'}:
                raise ValueError('Unexpected native ordinary screen metadata')
            native['state']['screen_observation']=copy.deepcopy(screen)
        if annotate_record(native,TRAILING_INPUT_VERSION)!=student:
            raise ValueError('Student request differs from current public formatter')
        if public_context(teacher['state'])['placement']!=rank or not public_context(teacher['state'])['late_match_window']:
            raise ValueError('Native current public placement differs')
        keys=original_policy_keys(teacher)
        if (len(keys)<2 or keys!=row['effective_action_keys'] or keys!=action_keys(student)
                or any(v['type']=='hora' for v in teacher['questions']['action']['criteria'].values())):
            raise ValueError('Native ordinary support or no-win condition differs')
        if set(student)!= {'state','questions'} or set(student['questions'])!={'action'}:
            raise ValueError('Unexpected native student payload fields')
        if set(student['questions']['action'])-{'type','instructions','criteria'}:
            raise ValueError('Teacher labels or future metadata entered student input')
    if counts!=Counter({'train:3':32,'train:4':32,'validation:3':8,'validation:4':8}):
        raise ValueError('Native context rank/split quotas changed')
    return rows


def native_record(row):
    record=copy.deepcopy(row['student_request'])
    record['questions']['action']['src']='replay_native'
    record['_meta']={'id':row['id'],'group_id':row['group_id'],'split':row['split'],
        'source':'native_context_retention','bucket':'replay:native_context','variant':'clean',
        'phase':row['phase'],'source_initial_placement':row['rank'],
        'label_type':'frozen_original_policy_pending','native_reference_pending':True,
        'teacher_request_sha256':row['teacher_request_sha256'],
        'student_request_sha256':row['student_request_sha256']}
    # No target or label until a verified frozen-v8 query supplies one.
    return record


def reference_preparation_only(config,counts):
    if (not config.get('internal_reference_preparation') or config.get('reference_regularization')
            or config.get('fit_registered') is not False):
        raise ValueError('Unlabeled native pairs may only be staged for frozen-reference preparation')
    total=sum(counts.values())
    return {k:n/total for k,n in counts.items()},{k:1. for k in counts}
