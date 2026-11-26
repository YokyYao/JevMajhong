"""Independent basic teacher families; never change or automatically fit a student."""
from collections import Counter
import os
from pathlib import Path
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import read, sha
from tools.curriculum_calls_post_reach_data_v1 import write
from tools import collect_curriculum_basic_teacher as original

CONFIG = ROOT/'configs/training/curriculum-basic-mortal-data-v2.json'
OUT = ROOT/'runs/curriculum-basic-teacher-data-v2'
PLAN = ROOT/'docs/experiments/plans/curriculum-basic-teacher-supplement-20261125.md'


def main():
    allocation = resource(); started = time.monotonic(); config = read(CONFIG)
    if (config['out'] != 'runs/curriculum-basic-teacher-data-v2/data'
            or config['seed_start'] != 2026600000 or config['driver_cpu_threads'] != 1
            or config['driver_gpu_memory_fraction'] != .15 or OUT.exists()):
        raise ValueError('Only the separate fixed256-family supplement may run once')
    families = original.sources(config); seeds = {r['seed'] for r in families}
    if Counter(r['split'] for r in families) != {'train': 205, 'validation': 51}:
        raise ValueError('Whole-family TRAIN205/VAL51 split must be frozen before generation')
    registered = ROOT/'runs/curriculum-ri-balanced-replay-fit-v2/fit-plan/registered.json'
    reg = read(registered)
    args = tomllib.loads(Path(reg['training_config']['trainer_config']).read_text())
    files = {str(p): sha(p) for p in (CONFIG, Path(__file__), Path(original.__file__), PLAN, registered)}
    inspected = 0
    import json
    for key in ('data', 'eval_data'):
        path = Path(args[key]); digest = sha(path)
        if reg['inputs'].get(str(path)) != digest:
            raise ValueError('Original frozen curriculum source changed')
        files[str(path)] = digest
        for line in path.open():
            meta = json.loads(line).get('_meta', {}); inspected += 1
            if any(meta.get(k) in seeds for k in ('source_logical_seed', 'logical_seed', 'source_seed')):
                raise ValueError('A current TRAIN/VAL source family overlaps the new supplement')
    if inspected != 87595+23688:
        raise ValueError('Full original TRAIN/VAL family metadata coverage required')
    import torch
    torch.set_num_threads(1)
    if torch.cuda.device_count() != 1:
        raise ValueError('Exactly one scheduler-allocated GPU required')
    free, total = torch.cuda.mem_get_info()
    if free < 20*1024**3:
        raise RuntimeError('Defer fresh collection rather than compete with the main fit for memory')
    torch.cuda.set_per_process_memory_fraction(config['driver_gpu_memory_fraction'], 0)
    # A step can expose its assigned GPU through STEP_GPUS. Propagate only that
    # real allocation to the unchanged source collector's legacy guard.
    if not os.environ.get('SLURM_JOB_GPUS'):
        if not os.environ.get('SLURM_STEP_GPUS'):
            raise ValueError('Actual GPU assignment environment required')
        os.environ['SLURM_JOB_GPUS'] = os.environ['SLURM_STEP_GPUS']
    OUT.mkdir()
    write(OUT/'driver-registered-private.json', dict(files=files, allocation=allocation,
        families=families, current_TRAIN_VAL_metadata_rows_checked=inspected,
        whole_family_split_before_generation=True, no_student_model_load_or_updates=True,
        original_collector_requested_threads=4, actual_driver_threads=1,
        GPU_memory_fraction=config['driver_gpu_memory_fraction'], initial_GPU_free_bytes=free,
        source_generation_no_error_or_outcome_filter=True, no_VAL_selection_for_TRAIN=True))
    original_set_threads = torch.set_num_threads
    def set_threads(requested):
        if requested != 4:
            raise ValueError('Review an unexpected changed original collector thread request')
        return original_set_threads(config['driver_cpu_threads'])
    torch.set_num_threads = set_threads
    try:
        original.collect(CONFIG)
    finally:
        torch.set_num_threads = original_set_threads
    summary = read(OUT/'data/summary-private.json')
    if (summary['stats']['complete_sources'] != 256
            or summary['registered_families'] != {'train': 205, 'validation': 51}
            or summary['student_optimizer_updates'] != 0 or summary['student_fit_admitted'] is not False
            or any(sha(path) != expected for path, expected in files.items())):
        raise ValueError('Full original collection, frozen sources and no-fit evidence required')
    for path in (OUT/'driver-registered-private.json', OUT/'data/registered-private.json',
                 OUT/'data/summary-private.json'):
        files[str(path)] = sha(path)
    write(OUT/'completion-binding-private.json', {'files': files})
    print(json.dumps(dict(status='independent_basic256_teacher_supplement_collected_not_fit_admitted',
        families=summary['registered_families'], coverage=summary['coverage'],
        no_student_updates=True, exact_native_encoder_ancestral_audit_still_required=True,
        seconds=time.monotonic()-started)), flush=True)


if __name__ == '__main__':
    main()
