"""Freeze an actually finished placement candidate before reserved outcomes.

This consumes complete final-source evidence; it does not produce missing
provenance, select checkpoints, submit games, or admit a partial course.
Run only inside an allocated, finite GPU job.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools import curriculum_confirmation_early_v1 as consumer
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_confirmation_contract_v1 import CRITERIA, require_criteria, require_final_parent
from tools.curriculum_transfer import canonical_protocol, read, sha

CRITERIA_PATH = ROOT/'configs/training/curriculum-confirmation-final32-v1.json'
OUT = ROOT/'runs/curriculum-final-confirmation-early-v1'
PUBLIC = ('tools/seal_curriculum_confirmation_early_v1.py',
    'configs/training/curriculum-confirmation-final32-v1.json',
    'tests/core/test_curriculum_confirmation_early_seal_v1.py',
    'docs/experiments/plans/curriculum-confirmation-early-seal-20261125.md')
SOURCE_STATUS = 'final_candidate_complete_lineages_source_catalog_and_seed_exclusion_audited'


def add_file(files, path, expected=None):
    path = Path(path).resolve(); key = str(path); actual = sha(path)
    if expected is not None and actual != expected:
        raise ValueError('A completed prerequisite changed: '+key)
    if key in files and files[key] != actual:
        raise ValueError('Conflicting final prerequisite identity: '+key)
    files[key] = actual


def add_receipt(files, path):
    path = Path(path).resolve()
    for name, digest in read(path)['files'].items():
        add_file(files, ROOT/name, digest)
    add_file(files, path)


def require_source_receipt(path, candidate, reference, files):
    """Consume a genuine producer receipt, never relabel a snapshot as final."""
    path = Path(path).resolve(); proof = read(path)
    if (proof.get('status') != SOURCE_STATUS
            or proof['candidate_identity'] != candidate['identity_sha256']
            or proof['reference_identity'] != reference['identity_sha256']
            or proof['seed_start'] != CRITERIA['seed_start']
            or proof['seed_groups'] != CRITERIA['seed_groups']
            or proof['reserved_source_data_hits'] != 0 or proof['unreviewed_artifacts']
            or proof['preliminary_probe_used_as_final_proof'] is not False
            or proof['reserved_games_generated'] != 0
            or not proof['producer_source_files'] or not proof['artifact_files']
            or not proof['json_artifact_classifications']):
        raise ValueError('Actual complete final-source proof required, not a snapshot or plan')
    receipt_path = path.parent/'completion-binding-private.json'
    receipt = read(receipt_path)['files']
    required = {str(path): sha(path)}
    for section in ('producer_source_files', 'artifact_files'):
        for name, digest in proof[section].items():
            key = str((ROOT/name).resolve())
            if name != key:
                raise ValueError('Final-source evidence requires canonical absolute paths')
            if key in required and required[key] != digest:
                raise ValueError('Conflicting producer/artifact evidence')
            required[key] = digest
    if any(receipt.get(name) != digest for name, digest in required.items()):
        raise ValueError('Actual final-source producer did not bind all its evidence')
    add_receipt(files, receipt_path)
    return proof


def require_anchor(path):
    """Require an already consumed, complete native development result."""
    path = Path(path).resolve()
    if path.name != 'matches-private.jsonl':
        raise ValueError('An actual consumed-development match file is required')
    anchor = None
    with path.open() as stream:
        for line in stream:
            if not line.strip(): continue
            row = json.loads(line)
            if row.get('profile') == 'efficiency' and row.get('seat') == 0:
                anchor = row
                break
    if (anchor is None or type(anchor['seed']) is not int
            or CRITERIA['seed_start'] <= anchor['seed'] < CRITERIA['seed_start']+32
            or anchor['completed'] is not True or anchor['ledger_passed'] is not True
            or not anchor['mjai_log']):
        raise ValueError('A complete non-reserved native reference anchor is required')
    return anchor


def make_registration(provenance_path, anchor_path):
    """No writes/model loads/games: derive the sole final seal from real inputs."""
    placement_path = consumer.course.REGISTERED.resolve()
    placement = read(placement_path)
    consumer.require_route(placement_path, placement)
    run = placement_path.parent.parent; fit = ROOT/placement['curriculum']['out']
    done_path = fit/'completion.json'; evidence_path = run/'final-evidence-private.json'
    audit_path = run/'audit-execution-private.json'
    admitted = require_final_parent(placement, read(done_path), read(evidence_path), read(audit_path))
    runtime = dict(source=placement['source'], source_sha256=placement['source_sha256'],
        upstream_source=placement['training_config']['upstream_source'],
        upstream_sha256=placement['training_config']['upstream_source_sha256'],
        environment_versions=consumer.original.environment_versions())
    sys.path[:0] = [runtime['source'], runtime['upstream_source']]
    from jevmahjong.evaluation.unified_contract import bundle
    from tools.curriculum_placement_parent_runtime_v1 import policy_contract
    candidate = bundle(fit/'training-run/checkpoints'/f"step-{admitted['selected_step']:06d}")
    reference = bundle(ROOT/placement['curriculum']['reference_checkpoint'])
    candidate_spec, candidate_id = policy_contract(candidate, placement)
    reference_spec, reference_id = consumer.original.reference_policy(reference, runtime)
    if (candidate['identity_sha256'] != admitted['identity']
            or candidate_id != admitted['policy_identity']
            or reference['identity_sha256'] != consumer.original.REFERENCE):
        raise ValueError('Only the actual audited final budget and frozen v8 are eligible')
    files = {}
    add_receipt(files, consumer.BINDING)
    add_receipt(files, run/'launch-binding-private.json')
    for name, digest in placement['inputs'].items(): add_file(files, ROOT/name, digest)
    require_source_receipt(provenance_path, candidate, reference, files)
    require_anchor(anchor_path)
    reference_protocol = Path(reference['directory']).parents[2]/'protocol.json'
    for protocol_path in (fit/'protocol.json', reference_protocol):
        protocol = canonical_protocol(read(protocol_path))
        for section in ('input_sha256', 'code_sha256'):
            for name, digest in protocol[section].items(): add_file(files, ROOT/name, digest)
        add_file(files, protocol_path)
    for model in (candidate, reference):
        for name, digest in model['files'].items(): add_file(files, Path(model['directory'])/name, digest)
    for path in (placement_path, done_path, evidence_path, audit_path, provenance_path,
            anchor_path, CRITERIA_PATH, *[ROOT/name for name in (*PUBLIC, *consumer.PUBLIC,
                *consumer.course.PUBLIC, *consumer.course.parent.PUBLIC)],
            *[ROOT/name for name in (
                'tools/run_curriculum_confirmation_v1.py', 'tools/curriculum_confirmation_runtime_v1.py',
                'tools/curriculum_confirmation_contract_v1.py', 'tools/curriculum_confirmation_native_v1.py',
                'tools/curriculum_confirmation_stats_v1.py', 'tools/play_curriculum_placement_v1.py',
                'tools/audit_curriculum_confirmation_v1.py', 'tools/finish_curriculum_confirmation_audit_v1.py',
                'cluster/curriculum-confirmation-run-v1.slurm', 'cluster/curriculum-confirmation-audit-v1.slurm',
                'tools/evaluate_curriculum_riichi_class_route_v2.py',
                'tools/curriculum_calls_post_reach_data_v1.py', 'tools/curriculum_transfer.py')]):
        add_file(files, path)
    require_criteria(read(CRITERIA_PATH))
    registration = dict(status=consumer.original.STATUS, criteria=copy.deepcopy(CRITERIA),
        criteria_config=str(CRITERIA_PATH.resolve()), placement_registration=str(placement_path),
        game_mode='4p-red-east', rule='GameRule.default_mjsoul', seed_namespace='spaced-v1',
        role_order=['reference', 'candidate'], candidate_model=candidate, reference_model=reference,
        candidate_policy=candidate_spec, candidate_policy_identity=candidate_id,
        reference_policy=reference_spec, reference_policy_identity=reference_id,
        runtime=runtime, provenance_evidence=str(Path(provenance_path).resolve()),
        reference_training_protocol=str(reference_protocol.resolve()),
        reference_development_anchor=str(Path(anchor_path).resolve()),
        out=str((OUT/'matches').resolve()), optimizer_updates_during_confirmation=0,
        reserved_grid_already_opened=False, automatic_release=False, allow_adaptive_extra_games=False,
        raw_matches_private=True, requires_independent_postrun_audit=True, files=files)
    consumer.verify(registration, consumer.REGISTERED)
    return registration


def seal(provenance_path, anchor_path):
    allocation = resource()
    if OUT.exists(): raise ValueError('Never overwrite an existing or partially written final seal')
    registration = make_registration(provenance_path, anchor_path)
    # Verification precedes all final-seal writes. A failed prerequisite leaves
    # no registration; a partial write remains preserved and cannot be retried.
    OUT.mkdir(parents=True, exist_ok=False)
    write(OUT/'sealer-execution-private.json', dict(allocation=allocation,
        reserved_games_generated=0, model_loaded=False, optimizer_updates=0,
        actual_final_candidate_identity=registration['candidate_model']['identity_sha256']))
    add_file(registration['files'], OUT/'sealer-execution-private.json')
    consumer.verify(registration, consumer.REGISTERED)
    write(consumer.REGISTERED, registration)
    print(json.dumps(dict(status=registration['status'], reserved_games_generated=0,
        final_registration_sha256=sha(consumer.REGISTERED), automatic_release=False)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--provenance', type=Path, required=True)
    parser.add_argument('--reference-development-anchor', type=Path, required=True)
    args = parser.parse_args()
    seal(args.provenance.resolve(), args.reference_development_anchor.resolve())
