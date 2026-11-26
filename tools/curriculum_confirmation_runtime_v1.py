"""Consumer contract for a future actual final seal, not a seal producer.

Existing statistical/training registrations and the preliminary source probe
cannot admit execution. A completed full-source registration is still needed.
"""
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
from tools.curriculum_transfer import read,sha,canonical_protocol
from tools.curriculum_confirmation_contract_v1 import CRITERIA,require_criteria,require_final_parent

STATUS='final32_actual_qualified_candidate_and_full_source_evidence_sealed_not_opened'
REFERENCE='5ebce6af3777854cbec55ba29db2dc90be0f959193114bb3154179c5552370fa'


def environment_versions():
    return {name:version(name) for name in ('torch','transformers','peft','riichienv')}


def reference_policy(identity,runtime):
    spec=dict(method='frozen_practical_v8_original_PolicyAgent',parameter_identity=identity['identity_sha256'],
        runtime_source_sha256=runtime['source_sha256'],upstream_source_sha256=runtime['upstream_sha256'],
        original_agent_sha256=sha(Path(runtime['source'])/'jevmahjong/evaluation/agents.py'),
        explicit_event_adapter_sha256=sha(ROOT/'tools/curriculum_confirmation_native_v1.py'),
        ordinary_guard_and_raw_argmax_unchanged=True,early_and_late_forced_legal_wins_unchanged=True)
    return spec,hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()


def require_header(reg):
    if (reg.get('status')!=STATUS or reg['criteria']!=CRITERIA or reg['game_mode']!='4p-red-east'
        or reg['rule']!='GameRule.default_mjsoul' or reg['seed_namespace']!='spaced-v1'
        or reg['role_order']!=['reference','candidate'] or reg['reference_model']['identity_sha256']!=REFERENCE
        or reg['optimizer_updates_during_confirmation']!=0 or reg['reserved_grid_already_opened']
        or reg['automatic_release'] or reg['allow_adaptive_extra_games']
        or not reg['raw_matches_private'] or not reg['requires_independent_postrun_audit']):
        raise ValueError('Actual sealed final32 protocol required; plans/partial training do not admit execution')
    require_criteria(reg['criteria'])


def cells(criteria):
    require_criteria(criteria)
    return [(seed,profile,seat) for seed in range(criteria['seed_start'],criteria['seed_start']+32)
        for profile in criteria['opponent_profiles'] for seat in criteria['seats']]


def require_output_isolation(directory,protected):
    directory=Path(directory).resolve()
    for value in protected:
        path=Path(value).resolve()
        if directory==path or path in directory.parents:
            raise ValueError('Confirmation outputs must be isolated from training/model/source directories')


def require_bound_files(files,receipt):
    for path,expected in receipt.items():
        if files.get(str(Path(path).resolve()))!=expected:
            raise ValueError('An actual ancestor/launch/source input escaped the final binding: '+path)


def verify(reg,registered):
    require_header(reg);registered=Path(registered).resolve()
    for path,expected in reg['files'].items():
        if sha(path)!=expected:raise ValueError('Final sealed input changed: '+path)
    require_criteria(read(reg['criteria_config']))
    runtime=reg['runtime'];sys.path.insert(0,runtime['source']);sys.path.insert(0,runtime['upstream_source'])
    from jevmahjong.training.base_lora import package_digest
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.initialization import lineage_training_files
    from tools.curriculum_placement_runtime_v1 import require_registration,policy_contract
    if (package_digest(Path(runtime['source'])/'jevmahjong')!=runtime['source_sha256']
        or package_digest(Path(runtime['upstream_source'])/'jevany')!=runtime['upstream_sha256']
        or environment_versions()!=runtime['environment_versions']):
        raise ValueError('Actual pinned runtime/environment differs from the final seal')
    placement_path=Path(reg['placement_registration']).resolve();placement=read(placement_path);require_registration(placement)
    run=placement_path.parent.parent;fit=ROOT/placement['curriculum']['out']
    require_bound_files(reg['files'],placement['inputs'])
    require_bound_files(reg['files'],read(run/'launch-binding-private.json')['files'])
    admitted=require_final_parent(placement,read(fit/'completion.json'),read(run/'final-evidence-private.json'),
        read(run/'audit-execution-private.json'))
    final=fit/'training-run/checkpoints'/f"step-{admitted['selected_step']:06d}"
    candidate=bundle(final);reference=bundle(ROOT/placement['curriculum']['reference_checkpoint'])
    candidate_spec,candidate_id=policy_contract(candidate,placement)
    reference_spec,reference_id=reference_policy(reference,runtime)
    if (candidate!=reg['candidate_model'] or reference!=reg['reference_model']
        or candidate['identity_sha256']!=admitted['identity'] or candidate_id!=admitted['policy_identity']
        or candidate_spec!=reg['candidate_policy'] or candidate_id!=reg['candidate_policy_identity']
        or reference_spec!=reg['reference_policy'] or reference_id!=reg['reference_policy_identity']
        or placement['source_sha256']!=runtime['source_sha256']
        or placement['training_config']['upstream_source_sha256']!=runtime['upstream_sha256']):
        raise ValueError('Actual qualified final candidate or original frozen-v8 policy differs')
    provenance_path=Path(reg['provenance_evidence']).resolve();provenance=read(provenance_path)
    if (provenance['status']!='final_candidate_complete_lineages_source_catalog_and_seed_exclusion_audited'
        or provenance['candidate_identity']!=candidate['identity_sha256']
        or provenance['reference_identity']!=reference['identity_sha256']
        or provenance['seed_start']!=2026300000 or provenance['seed_groups']!=32
        or provenance['reserved_source_data_hits']!=0 or provenance['unreviewed_artifacts']
        or provenance['preliminary_probe_used_as_final_proof'] or provenance['reserved_games_generated']!=0):
        raise ValueError('Complete post-freeze source evidence required; preliminary JSONL probe is insufficient')
    # These receipts must come from the actual future full-source producer and
    # remain bound. This consumer does not invent missing classification work.
    if (not provenance['producer_source_files'] or not provenance['artifact_files']
        or not provenance['json_artifact_classifications']):
        raise ValueError('Source producer, actual artifacts and JSON classification evidence required')
    if any(path in provenance['artifact_files'] and provenance['artifact_files'][path]!=expected
        for path,expected in provenance['producer_source_files'].items()):
        raise ValueError('Conflicting source producer/artifact evidence')
    for path,expected in {**provenance['producer_source_files'],**provenance['artifact_files']}.items():
        if reg['files'].get(path)!=expected or sha(path)!=expected:
            raise ValueError('Actual classified source artifact/producer is unbound or changed')
    reference_protocol=Path(reg['reference_training_protocol']).resolve()
    expected_reference_protocol=Path(reference['directory']).resolve().parents[2]/'protocol.json'
    if reference_protocol!=expected_reference_protocol:
        raise ValueError('Original frozen-v8 training protocol required')
    protocols=dict(candidate=canonical_protocol(read(fit/'protocol.json')),
        reference=canonical_protocol(read(reference_protocol)))
    if any(protocols['reference']['config'][key]!=placement['training_config'][key] for key in ('base','base_revision')):
        raise ValueError('Candidate and frozen reference require the same pinned pretrained base')
    for role,protocol in protocols.items():
        require_bound_files(reg['files'],protocol['input_sha256'])
        require_bound_files(reg['files'],protocol['code_sha256'])
        actual=lineage_training_files(protocol)
        if actual!=provenance['training_lineages'][role]:raise ValueError('Full actual model training ancestry differs')
        required=set(actual)|{protocol['trainer_args']['eval_data']}
        required|={str(Path(p).resolve()) for p in protocol['input_sha256'] if Path(p).suffix=='.jsonl'}
        if not required<=set(provenance['artifact_files']):raise ValueError('An actual ancestral data input escaped source checking')
    required_paths=[registered,placement_path,fit/'completion.json',run/'final-evidence-private.json',
        run/'audit-execution-private.json',fit/'protocol.json',reference_protocol,provenance_path,Path(reg['criteria_config']),
        ROOT/'tools/run_curriculum_confirmation_v1.py',ROOT/'tools/curriculum_confirmation_runtime_v1.py',
        ROOT/'tools/curriculum_confirmation_contract_v1.py',ROOT/'tools/curriculum_confirmation_native_v1.py',
        ROOT/'tools/curriculum_confirmation_stats_v1.py',ROOT/'tools/play_curriculum_placement_v1.py',
        ROOT/'tools/audit_curriculum_confirmation_v1.py',ROOT/'tools/finish_curriculum_confirmation_audit_v1.py',
        ROOT/'cluster/curriculum-confirmation-run-v1.slurm',ROOT/'cluster/curriculum-confirmation-audit-v1.slurm',
        ROOT/'tools/evaluate_curriculum_riichi_class_route_v2.py',
        ROOT/'tools/curriculum_calls_post_reach_data_v1.py',ROOT/'tools/curriculum_transfer.py']
    # The registration cannot hash itself; the runner records its actual hash
    # before the first outcome and checks it again at the end.
    if any(str(path.resolve()) not in reg['files'] for path in required_paths if path!=registered):
        raise ValueError('Final execution/evidence implementation missing from the pre-outcome binding')
    anchor=Path(reg['reference_development_anchor']).resolve()
    if str(anchor) not in reg['files'] or not str(anchor).endswith('matches-private.jsonl'):
        raise ValueError('Actual consumed-development v8 anchor must be bound before reserved games')
    return dict(placement=placement,candidate=candidate,reference=reference,
        candidate_policy=candidate_spec,reference_policy=reference_spec)
