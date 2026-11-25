"""Finite exact native/encoder audit plus full current-course split exclusions."""
from collections import defaultdict
import json
import os
from pathlib import Path
import sys
import time
import tomllib
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools import audit_curriculum_basic_teacher as original
from tools.audit_curriculum_basic_teacher import fingerprint

allocation = resource(); started = time.monotonic()
root = ROOT/'runs/curriculum-basic-teacher-data-v2'
out = root/'audit-v1'
driver_receipt = root/'audit-driver-registered-private.json'
if out.exists() or driver_receipt.exists():
    raise ValueError('Never overwrite original or failed audit evidence')
binding = read(root/'completion-binding-private.json')['files']
if any(sha(path) != expected for path, expected in binding.items()):
    raise ValueError('Only unchanged completed independent collection is auditable')
source = read(root/'data/summary-private.json')
if (source['stats']['complete_sources'] != 256 or source['registered_families'] != {'train':205,'validation':51}
        or source['student_fit_admitted'] is not False):
    raise ValueError('All256 prospective whole-family sources must be complete')
registered = ROOT/'runs/curriculum-ri-balanced-replay-fit-v2/fit-plan/registered.json'
reg = read(registered); args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
files = dict(binding)
for path in (registered, Path(__file__), Path(original.__file__), root/'completion-binding-private.json'):
    files[str(path)] = sha(path)
for key in ('data','eval_data'):
    path = Path(args[key]); digest = sha(path)
    if reg['inputs'].get(str(path)) != digest:
        raise ValueError('Complete current TRAIN/VAL source changed')
    files[str(path)] = digest
import torch
torch.set_num_threads(1)
if torch.cuda.device_count() != 1:
    raise ValueError('One actual scheduler GPU required even for replay/encoder data processing')
if torch.cuda.mem_get_info()[0] < 20*1024**3:
    raise RuntimeError('Defer audit instead of competing with the active fit for memory')
torch.cuda.set_per_process_memory_fraction(.15,0)
if not os.environ.get('SLURM_JOB_GPUS'):
    if not os.environ.get('SLURM_STEP_GPUS'):
        raise ValueError('Actual GPU assignment required')
    os.environ['SLURM_JOB_GPUS'] = os.environ['SLURM_STEP_GPUS']
write(driver_receipt,dict(files=files,allocation=allocation,actual_threads=1,
    original_auditor_requested_threads=4,no_model_or_student_updates=True,
    full_current_course_exclusions_required_after_original_native_encoder_audit=True))
old_threads = torch.set_num_threads
def cap_threads(requested):
    if requested != 4:
        raise ValueError('Review changed original auditor thread request')
    return old_threads(1)
torch.set_num_threads = cap_threads
try:
    original.audit(root/'data',out)
finally:
    torch.set_num_threads = old_threads
audited = read(out/'summary-private.json')
if (audited['checked_complete_source_games'] != 256 or audited['student_fit_admitted'] is not False
        or not audited['source_labels_unchanged'] or not audited['source_splits_unchanged']):
    raise ValueError('Full original native/encoder audit required')
current = {}
for split,key,expected in (('train','data',87595),('validation','eval_data',23688)):
    sets = {k:set() for k in ('ids','groups','inputs')}; count = 0
    for line in Path(args[key]).open():
        row = json.loads(line); meta = row['_meta']; count += 1
        sets['ids'].add(meta['id']); sets['groups'].add(meta['group_id']); sets['inputs'].add(fingerprint(row))
    if count != expected or len(sets['ids']) != count:
        raise ValueError('Full current-course population required')
    current[split] = sets
all_current_ids = current['train']['ids'] | current['validation']['ids']
all_current_groups = current['train']['groups'] | current['validation']['groups']
overlaps = {}; normalized_counts = {}
for split,other in (('train','validation'),('validation','train')):
    spec = audited['files'][split]; path = ROOT/spec['path']
    if sha(path) != spec['sha256']:
        raise ValueError('Normalized teacher output changed')
    count = 0; collisions = defaultdict(int)
    for line in path.open():
        row = json.loads(line); meta = row['_meta']; count += 1; key = fingerprint(row)
        collisions['same_split_public_input'] += key in current[split]['inputs']
        collisions['opposite_split_public_input'] += key in current[other]['inputs']
        collisions['any_current_id'] += meta['id'] in all_current_ids
        collisions['any_current_group'] += meta['group_id'] in all_current_groups
    if count != spec['records'] or any(collisions.values()):
        raise ValueError('Fresh sources must not duplicate current TRAIN/VAL inputs, IDs or groups')
    overlaps[split] = dict(collisions); normalized_counts[split] = count
    files[str(path)] = spec['sha256']
if any(sha(path) != expected for path, expected in files.items()):
    raise ValueError('Prospective collection/audit sources changed')
summary = dict(status='independent_basic256_native_encoder_and_full_current_course_exclusions_passed_not_fit_admitted',
    families=source['registered_families'],normalized_counts=normalized_counts,
    classes=audited['counts'], checked_complete_source_games=256,
    exact_native_source_labels_and_splits_unchanged=True,full_current_TRAIN_VAL_overlap_counts=overlaps,
    no_model_or_student_updates=True,no_new_native_matches_in_audit=True,
    no_reserved_final_payload_read=True,student_fit_admitted=False,
    seconds=time.monotonic()-started,allocation=allocation)
write(root/'audit-driver-summary-private.json',summary)
for path in (driver_receipt,out/'summary-private.json',out/'audit-binding-private.json',root/'audit-driver-summary-private.json'):
    files[str(path)] = sha(path)
write(root/'audit-driver-completion-binding-private.json',{'files':files})
print(json.dumps({k:v for k,v in summary.items() if k!='allocation'}),flush=True)
