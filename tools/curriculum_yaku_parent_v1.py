"""Seal the actual audited final 3A parameters for the fixed 3B course."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.curriculum_transfer import read, sha, canonical_protocol
from tools.curriculum_parent import rows, require_replay
from tools.curriculum_calls_post_reach_data_v1 import gpu_only, write
from tools.curriculum_progression_parent_v1 import extend
from tools.seal_curriculum_basic_repair_v2 import scheduler_exit

METHOD = 'audited_final_progression2400_same_policy_yaku_continuation_v1'
STATUS = 'actual_progression2400_final_sealed_for_yaku_course'


def seal(config, out):
    gpu_only(); c = read(config)
    if (c['parent_selection'] != 'actual_registered_final2400_only'
            or c['admission_mode'] != 'original_stage3A_grade_pass_required'):
        raise ValueError('The actual qualified final 3A is required; no peak selection')
    registered = ROOT/c['parent_registration']; parent = read(registered); pc = parent['curriculum']
    fit = ROOT/pc['out']; run = registered.parent.parent
    evidence_path = ROOT/c['parent_evidence']; evidence = read(evidence_path); done = read(fit/'completion.json')
    execution = read(run/'audit-execution-private.json')
    if (parent['registered_final_step'] != 2400
            or done['status'] != 'progression2400_fit_and_real_curves_complete'
            or done['optimizer_steps'] != 2400 or done['actual_records_seen'] != 175190
            or evidence['status'] != 'progression2400_final_evidence_audit_passed'
            or evidence['stage3_advancement']['passed'] is not True
            or evidence['saved_final_reloaded_menus'] != 23688
            or execution['status'] != 'actual_progression2400_full_evidence_independently_audited'
            or not execution['original_launch_binding_unchanged']):
        raise ValueError('Full audited 3A budget and original advancement grade required')
    sys.path.insert(0, parent['source']); sys.path.insert(0, parent['training_config']['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.initialization import lineage_training_files, check_groups
    if package_digest(Path(parent['source'])/'jevmahjong') != parent['source_sha256']:
        raise ValueError('Parent pinned runtime changed')
    files = dict(parent['inputs'])
    binding = read(run/'launch-binding-private.json')
    files.update(binding['files'])
    captured = read(run/'fit-terminal-scheduler-private.json')
    exits = dict(fit=scheduler_exit(captured['exit'], captured['raw'], ROOT/'cluster/curriculum-progression-v1.slurm'))
    receipt_path = ROOT/'logs'/f"curriculum-progression-exit-{c['parent_audit_job']}.json"
    receipt = read(receipt_path)
    if receipt['phase'] != 'audit':
        raise ValueError('Actual 3A audit phase receipt required')
    archive = run/'audit-terminal-for-yaku-private.json'
    if archive.exists():
        archived = read(archive)
        if archived['exit'] != receipt:
            raise ValueError('Archived audit receipt differs')
        raw = archived['raw']
    else:
        raw = subprocess.run(['scontrol', '-o', 'show', 'job', str(c['parent_audit_job'])],
            check=True, capture_output=True, text=True, timeout=20).stdout
        scheduler_exit(receipt, raw, ROOT/'cluster/curriculum-progression-v1.slurm')
        write(archive, dict(raw=raw, exit=receipt))
    exits['audit'] = scheduler_exit(receipt, raw, ROOT/'cluster/curriculum-progression-v1.slurm')
    if str(captured['exit']['job_id']) != str(c['parent_fit_job']):
        raise ValueError('Unexpected 3A fit job')
    data_path = ROOT/c['data_ready']; data = read(data_path)
    if (data['status'] != 'exact_replay_and_normalized_targeted_yaku_corpus_prepared_not_fit_admitted'
            or not data['original_replay_objects_labels_and_partitions_unchanged']
            or data['cross_split_family_and_public_input_overlap'] != 0):
        raise ValueError('Exact audited replay/yaku data required')
    recipe = ROOT/c['objective_preflight']
    extend(files, data_path.parent/'completion-binding-private.json')
    extend(files, recipe.parent/'completion-binding-private.json')
    args = tomllib.loads(Path(parent['training_config']['trainer_config']).read_text())
    for split, key, size in (('train', 'data', 91247), ('validation', 'eval_data', 24516)):
        path = ROOT/data['files'][split]['path']
        if sha(path) != data['files'][split]['sha256'] or data['files'][split]['records'] != size:
            raise ValueError('Child data changed')
        require_replay(rows(args[key]), rows(path)); files[str(path.resolve())] = sha(path)
    protocol = canonical_protocol(read(fit/'protocol.json'))
    lineage = lineage_training_files(protocol)
    groups = check_groups(args['data'], ROOT/data['files']['train']['path'],
        ROOT/data['files']['validation']['path'], lineage[1:])
    model = bundle(fit/'training-run/checkpoints/step-002400')
    contract = read(fit/'policy-contracts-private.json')['2400']
    if (model['identity_sha256'] != evidence['final_identity']
            or contract['policy_identity'] != evidence['final_policy_identity']
            or contract['model_identity'] != model['identity_sha256']):
        raise ValueError('Exact audited final parameters and unchanged policy required')
    cache = fit/'evaluations/step-002400'; summary = read(cache/'summary.json')
    if (len(rows(cache/'matches-private.jsonl')) != 64
            or summary['checkpoint_sha256'] != model['identity_sha256']
            or summary['policy_identity'] != contract['policy_identity']):
        raise ValueError('Reuse actual full final64 parent cache')
    out.mkdir(parents=True, exist_ok=False)
    paths = [Path(__file__), config, registered, evidence_path, archive, receipt_path, data_path, recipe,
        run/'launch-binding-private.json', run/'fit-terminal-scheduler-private.json', run/'audit-execution-private.json',
        fit/'completion.json', fit/'protocol.json', fit/'policy-contracts-private.json']
    paths += [fit/'metrics'/name for name in ('train_metrics.jsonl', 'validation_metrics.jsonl', 'evaluation_metrics.jsonl')]
    paths += [cache/name for name in ('protocol.json', 'summary.json', 'matches-private.jsonl')]
    paths += [Path(model['directory'])/name for name in model['files']]
    paths += [fit/'training-run/training_eval/step-002400/development/rows.json']
    for path in paths: files[str(path.resolve())] = sha(path)
    if any(sha(p) != h for p, h in files.items()):
        raise ValueError('An immutable parent/data prerequisite changed')
    result = dict(status=STATUS, method=METHOD, selected_step=2400, model=model,
        policy=contract['policy'], policy_identity=contract['policy_identity'],
        original_parent_grade=evidence['stage3_advancement'], parent_registration=str(registered.resolve()),
        parent_evidence=str(evidence_path.resolve()), parent_training_file=args['data'], parent_validation_file=args['eval_data'],
        parent_evaluation_cache=str(cache.resolve()), original_guarded_evaluation_cache=pc['original_guarded_evaluation_cache'],
        original_guarded_identity=pc['parent_identity'], child_data_ready=str(data_path.resolve()),
        base=pc['base'], base_revision=pc['base_revision'], bound_files=files, group_proof=groups,
        scheduler_exits=exits, raw_grade_overridden=False, optimizer_resumed=False,
        selection_from_minimum_CE=False, child_fit_started=False)
    write(out/'parent-seal-private.json', result)


def verify_parent(cfg):
    spec = cfg['parent_initialization']; path = Path(spec['manifest']).resolve(); sealed = read(path)
    if (cfg.get('initialization') != 'continued' or spec['method'] != METHOD
            or sha(path) != spec['sha256'] or sealed['method'] != METHOD or sealed['status'] != STATUS
            or spec['selected_step'] != 2400 or sealed['selected_step'] != 2400
            or sealed['original_parent_grade']['passed'] is not True or sealed['raw_grade_overridden']
            or sealed['optimizer_resumed'] or sealed['selection_from_minimum_CE']
            or sealed['model']['identity_sha256'] != spec['identity_sha256']
            or sealed['policy_identity'] != spec['policy_identity_sha256']
            or cfg['base'] != sealed['base'] or cfg['base_revision'] != sealed['base_revision']):
        raise ValueError('Exact original-qualified final 3A ancestry required')
    for p, h in sealed['bound_files'].items():
        if sha(p) != h: raise ValueError('Sealed prerequisite changed: '+p)
    parent = read(sealed['parent_registration']); fit = ROOT/parent['curriculum']['out']
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    if (Path(spec['source']).resolve() != Path(parent['source']).resolve()
            or spec['source_sha256'] != parent['source_sha256']
            or package_digest(Path(parent['source'])/'jevmahjong') != parent['source_sha256']
            or bundle(sealed['model']['directory']) != sealed['model']
            or Path(cfg['out']).resolve() == fit.resolve() or Path(spec['run']).resolve() != fit.resolve()
            or spec['child_data_ready'] != sealed['child_data_ready']):
        raise ValueError('Pinned parent model/runtime or independent output differs')
    args = tomllib.loads(Path(cfg['trainer_config']).read_text()); data = read(sealed['child_data_ready'])
    if args.get('resume') or Path(args['init_from']).resolve() != Path(sealed['model']['directory']).resolve():
        raise ValueError('Same adapter/head with a new optimizer required')
    for split, key in (('train', 'data'), ('validation', 'eval_data')):
        if Path(args[key]).resolve() != (ROOT/data['files'][split]['path']).resolve():
            raise ValueError('Child dataset differs from seal')
    protocol = canonical_protocol(read(fit/'protocol.json'))
    if protocol['config'] != parent['training_config']:
        raise ValueError('Parent protocol differs')
    receipt = dict(sealed['bound_files']); receipt[str(path)] = spec['sha256']
    return dict(model=sealed['model'], receipt=receipt, training_protocol=protocol,
        training_config=parent['training_config'], selected_step=2400)


@contextmanager
def parent_loader(initialization):
    original = initialization.verified_parent; initialization.verified_parent = verify_parent
    try: yield
    finally: initialization.verified_parent = original

