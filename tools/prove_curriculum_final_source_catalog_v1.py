"""Join real completed source proofs for the sole qualified placement model.

No checkpoint selection, model inference, wall resets, or final-test opening.
Known raw values retain their original evidence; only new metadata and hash
edges are observed here. Missing or unfamiliar inputs fail closed, with no
receipt claiming completion. Execute only in finite scheduler GPU resources.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from tools.capture_curriculum_paired_terminal_v2 import resource
from tools.curriculum_transfer import sha, canonical_protocol
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_confirmation_contract_v1 import CRITERIA, require_final_parent
from tools.inventory_curriculum_source_delta_v1 import Inventory
from tools.observe_curriculum_metadata_delta_v1 import decode, findings, observe_file, MAX_BYTES

BASE = ROOT/'runs/curriculum-placement-base-source-closure-v1'
TARGETED = ROOT/'runs/curriculum-complete-targeted-source-population-v2'
ORIGINS = ROOT/'runs/curriculum-placement-complete-data-origin-proof-v1'
METADATA = ROOT/'runs/curriculum-final3b-metadata-observations-v1'
DEFAULT_OUT = ROOT/'runs/curriculum-final-placement-source-proof-v1'
STATUS = 'final_candidate_complete_lineages_source_catalog_and_seed_exclusion_audited'


def merge_identities(destination, values):
    for name, expected in values.items():
        path = str((ROOT/name).resolve())
        if path in destination and destination[path] != expected:
            raise ValueError('Conflicting completed source identity: ' + path)
        destination[path] = expected


def add_role(catalogue, path, expected, role, evidence):
    path = str(Path(path).resolve())
    if path in catalogue:
        if catalogue[path]['sha256'] != expected:
            raise ValueError('Conflicting actual classified source: ' + path)
        return
    catalogue[path] = dict(path=path, sha256=expected, role=role, evidence=str(evidence))


def require_lineage_coverage(protocols, lineages, catalogue):
    for role, protocol in protocols.items():
        required = {str(Path(name).resolve()) for name in lineages[role]}
        required.add(str(Path(protocol['trainer_args']['eval_data']).resolve()))
        required.update(str(Path(name).resolve()) for name in protocol['input_sha256']
            if Path(name).suffix == '.jsonl')
        for name in required:
            expected = protocol['input_sha256'].get(name)
            if expected is None or name not in catalogue or catalogue[name]['sha256'] != expected:
                raise ValueError('An actual ancestral TRAIN/validation input lacks source evidence: ' + name)


class Evidence:
    """Read only actual hash-bound completed evidence, rejecting conflicts."""
    def __init__(self):
        self.files = {}; self.checked = {}

    def binding(self, directory):
        path = Path(directory).resolve()/'completion-binding-private.json'
        raw = path.read_bytes(); value = decode(raw)
        merge_identities(self.files, value['files'])
        merge_identities(self.files, {str(path): hashlib.sha256(raw).hexdigest()})
        return path

    def read(self, path):
        path = Path(path).resolve(); raw = path.read_bytes()
        actual = hashlib.sha256(raw).hexdigest()
        if self.files.get(str(path)) != actual:
            raise ValueError('Actual completed source evidence changed: ' + str(path))
        self.checked[str(path)] = actual
        return decode(raw)


def prior_catalogue(evidence):
    """Require the actual complete parents, never promote a subset proof."""
    for directory in (BASE, TARGETED, ORIGINS, METADATA):
        evidence.binding(directory)
    base = evidence.read(BASE/'summary-private.json')
    target = evidence.read(TARGETED/'summary-private.json')
    origins = evidence.read(ORIGINS/'summary-private.json')
    metadata = evidence.read(METADATA/'summary-private.json')
    if (base['status'] != 'completed_placement_base_sources_and_families_joined_not_final_candidate'
            or base['reserved_source_data_hits'] or base['unreviewed_artifacts']
            or target['status'] != 'actual_complete_targeted108_case_cache_values_and_native_seed_parameters_observed'
            or target['selected_views'] != 108 or not target['full108_views_actually_merged']
            or target['unreviewed_numeric_contexts']
            or target['actual_complete_case_files'] + target['quarantined_unadmitted_views'] != 108
            or origins['status'] != 'actual_complete_placement_parent_projection_and_all_virtual_exposure_origins_proved'
            or not origins['all_parent_rows_accounted_exactly_once']
            or not origins['all_inputs_targets_sources_and_candidate_order_preserved']
            or not origins['all_virtual_copies_change_only_original_id_and_explicit_trace']
            or origins['validation_copies'] or origins['unmatched_rows'] or origins['cross_split_source_families']
            or metadata['status'] != 'actual_new_bounded_metadata_delta_values_observed_not_final_source_proof'
            or metadata['reserved_source_data_hits'] or metadata['unreviewed_artifacts']):
        raise ValueError('Actual complete source, native-seed and row-origin evidence is required')
    catalogue = {}
    original = evidence.read(BASE/'actual-artifact-role-catalogue-private.json')
    assets = evidence.read(BASE/'actual-artifact-files-private.json')
    if len(assets) != base['actual_conservative_ancestral_and_prepared_base_artifacts']:
        raise ValueError('Completed base source count differs')
    for row in original:
        if assets.get(row['path']) != row['sha256'] or row['path'] in catalogue:
            raise ValueError('Original source role catalogue differs from completed identities')
        add_role(catalogue, row['path'], row['sha256'], row['role'], row['evidence'])
    if set(catalogue) != set(assets):
        raise ValueError('Every closed base input requires its actual role')
    path = TARGETED/'per-source-observations-private.jsonl'
    if evidence.files.get(str(path)) != sha(path):
        raise ValueError('Actual complete targeted source observations changed')
    evidence.checked[str(path)] = evidence.files[str(path)]
    seen = set(); cases = 0
    with path.open() as stream:
        for line in stream:
            if not line.strip(): continue
            row = decode(line); name = row['path']
            if (name in seen or not row['stable_during_read']
                    or evidence.files.get(name) != row['sha256']
                    or row['counts'].get('logical_seed_occurrences', 0)
                    or row['counts'].get('reserved_spaced_hand_stream_occurrences', 0)
                    or 'actual_' not in row['role']):
                raise ValueError('Every actual targeted source must retain clean bound value evidence')
            seen.add(name); cases += row['role'] == 'actual_complete_targeted_case_with_bound_source_family_and_derived_future_native_seeds'
            add_role(catalogue, name, row['sha256'], row['role'], path)
    if len(seen) != target['actual_disjoint_source_files'] or cases != target['actual_complete_case_files']:
        raise ValueError('Actual full targeted case/cache coverage differs')
    for row in evidence.read(METADATA/'per-source-observations-private.json'):
        if not row['stable_during_read'] or row['unreviewed_reserved_occurrences']:
            raise ValueError('Actual new metadata still needs context review')
        if evidence.files.get(row['path']) != row['sha256']:
            raise ValueError('New metadata observation is not bound')
        add_role(catalogue, row['path'], row['sha256'], row['role'], METADATA/'completion-binding-private.json')
    # The row-origin proof covers every target and field, not just ID membership.
    from tools.prove_curriculum_placement_complete_data_origins_v1 import EXPOSURES, COHERENT, TARGETED as TARGET_READY
    for ready_path in (EXPOSURES, COHERENT, TARGET_READY):
        ready = evidence.read(ready_path)
        for spec in ready['files'].values():
            name = str((ROOT/spec['path']).resolve())
            if evidence.files.get(name) != spec['sha256']:
                raise ValueError('Completed projected dataset escaped the row-origin binding')
            add_role(catalogue, name, spec['sha256'],
                'actual_complete_dataset_exact_projected_parent_or_explicit_virtual_exposure',
                ORIGINS/'completion-binding-private.json')
    expanded = evidence.read(EXPOSURES)
    if (origins['expanded_train_sha256'] != expanded['files']['train']['sha256']
            or origins['unique_train_records'] != expanded['unique_train_records']
            or origins['training_exposures_per_pass'] != expanded['training_exposures_per_pass']):
        raise ValueError('Final source proof must cover the exact actual fit exposure file')
    return catalogue, seen


def close_graph(graph, catalogue, edge_only, parameter_files):
    """Close new hash edges; unknown data/binary inputs are never guessed safe."""
    parsed = set(); observations = []; edges_only_count = 0
    while True:
        pending = [(name, node) for name, node in graph.nodes.items()
            if name not in parsed and (name not in catalogue or name in edge_only)]
        if not pending: break
        for name, node in pending:
            parsed.add(name); path = Path(name); suffix = path.suffix
            if suffix == '.json':
                before = path.stat(); raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != node['sha256']:
                    raise ValueError('A declared source changed while closing its hash edges')
                value = decode(raw); graph.walk(value, name)
                after = path.stat()
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise ValueError('Never observe a changing source graph')
                if name in edge_only:
                    # Actual typed values/native continuations are already proved.
                    edges_only_count += 1
                else:
                    if len(raw) > MAX_BYTES:
                        raise ValueError('New large JSON requires its own typed observer: ' + name)
                    row = observe_file(path, node['sha256'])
                    if row['unreviewed_reserved_occurrences']:
                        raise ValueError('Review actual new metadata seed context: ' + name)
                    observations.append(row)
                    add_role(catalogue, name, node['sha256'], row['role'], 'new_metadata_observed_in_final_source_producer')
            elif name in parameter_files:
                if parameter_files[name] != node['sha256'] or sha(path) != node['sha256']:
                    raise ValueError('Actual complete model parameter member changed')
                add_role(catalogue, name, node['sha256'],
                    'actual_complete_ancestral_model_parameter_member_not_source_rollout',
                    'qualified_model_or_actual_parent_bundle')
            elif suffix in ('.py', '.slurm', '.md', '.toml', '.jinja', '.txt', '.log'):
                raw = path.read_bytes()
                if hashlib.sha256(raw).hexdigest() != node['sha256']:
                    raise ValueError('Actual declared source text changed')
                _, hits = findings(raw.decode())
                if suffix in ('.txt', '.log') and hits:
                    raise ValueError('An actual transcript requires explicit seed-context review')
                add_role(catalogue, name, node['sha256'],
                    'actual_declared_executable_or_configuration_text' if suffix not in ('.md','.txt','.log')
                    else 'actual_declared_documentation_or_clean_transcript',
                    'exact_training_or_parent_hash_edge')
            else:
                raise ValueError('No completed typed source evidence for a newly declared input: ' + name)
        if len(graph.nodes) > 20000:
            raise ValueError('Review an unexpectedly expansive final source graph')
    if graph.issues:
        raise ValueError('Actual source graph has missing/conflicting declarations: ' + str(graph.issues[:3]))
    if any(name not in catalogue or catalogue[name]['sha256'] != node['sha256'] for name,node in graph.nodes.items()):
        raise ValueError('A declared source escaped its actual role catalogue')
    return observations, edges_only_count


def prove(out=DEFAULT_OUT):
    allocation = resource(); started = time.monotonic(); out = Path(out).resolve()
    if out.exists(): raise ValueError('Never overwrite complete or partial final source evidence')
    # Require the actual final full-budget grade and audit before any output.
    from tools import curriculum_confirmation_early_v1 as consumer
    placement_path = consumer.course.REGISTERED.resolve()
    placement = decode(placement_path.read_bytes()); consumer.require_route(placement_path, placement)
    run = placement_path.parent.parent; fit = ROOT/placement['curriculum']['out']
    admitted = require_final_parent(placement, decode((fit/'completion.json').read_bytes()),
        decode((run/'final-evidence-private.json').read_bytes()),
        decode((run/'audit-execution-private.json').read_bytes()))
    if consumer.REGISTERED.exists():
        raise ValueError('Complete actual source evidence before the sole final seal is created')
    sys.path[:0] = [placement['source'], placement['training_config']['upstream_source']]
    from jevmahjong.evaluation.unified_contract import bundle
    from jevmahjong.training.initialization import lineage_training_files
    for module in ('jevmahjong.evaluation.unified_contract', 'jevmahjong.training.initialization'):
        if not Path(sys.modules[module].__file__).resolve().is_relative_to(Path(placement['source']).resolve()):
            raise ValueError('The actual frozen lineage and bundle implementations are required')
    candidate = bundle(fit/'training-run/checkpoints'/f"step-{admitted['selected_step']:06d}")
    reference = bundle(ROOT/placement['curriculum']['reference_checkpoint'])
    if candidate['identity_sha256'] != admitted['identity'] or reference['identity_sha256'] != consumer.original.REFERENCE:
        raise ValueError('Only the actual qualified final candidate and frozen v8 are eligible')
    protocol_paths = dict(candidate=fit/'protocol.json', reference=Path(reference['directory']).parents[2]/'protocol.json')
    protocols = {role:canonical_protocol(decode(path.read_bytes())) for role,path in protocol_paths.items()}
    if any(protocols['reference']['config'][key] != placement['training_config'][key] for key in ('base','base_revision')):
        raise ValueError('The final candidate and v8 must use the same pinned pretrained base')
    lineages = {role:lineage_training_files(protocol) for role,protocol in protocols.items()}
    evidence = Evidence(); catalogue, edge_only = prior_catalogue(evidence)
    history = evidence.read(ROOT/'runs/curriculum-active-student-generic-config-history-v3/summary-private.json')
    graph = Inventory({name:row['sha256'] for name,row in catalogue.items()}, history)
    parameters = {}; roots = {}
    for model in (candidate, reference):
        merge_identities(parameters, {str(Path(model['directory'])/name):expected for name,expected in model['files'].items()})
    for protocol in protocols.values():
        parent = protocol.get('parent_initialization', {}).get('model')
        if parent is not None:
            if bundle(parent['directory']) != parent:
                raise ValueError('Actual immediate parent model bundle changed')
            merge_identities(parameters, {str(Path(parent['directory'])/name):expected for name,expected in parent['files'].items()})
        merge_identities(roots, protocol['input_sha256']); merge_identities(roots, protocol['code_sha256'])
    merge_identities(roots, placement['inputs'])
    for name,path in protocol_paths.items(): roots[str(path.resolve())] = sha(path)
    for path in (placement_path, run/'launch-binding-private.json', fit/'completion.json',
            run/'final-evidence-private.json', run/'audit-execution-private.json'):
        roots[str(path.resolve())] = sha(path)
    merge_identities(roots, parameters)
    # Include all conservative original sources and all actual new complete
    # case/cache sources. Only their newly introduced hash edges are read again.
    for name in edge_only: roots[name] = catalogue[name]['sha256']
    for name,expected in roots.items(): graph.add(name, expected, name, '/actual_final_root')
    observations, edge_count = close_graph(graph, catalogue, edge_only, parameters)
    require_lineage_coverage(protocols, lineages, catalogue)
    artifacts = {name:row['sha256'] for name,row in catalogue.items()}
    for name,expected in evidence.checked.items():
        if sha(name) != expected: raise ValueError('Actual completed evidence changed during the join')
    producer = {str(Path(__file__).resolve()):sha(__file__)}
    for path in (ROOT/'tools/inventory_curriculum_source_delta_v1.py',
            ROOT/'tools/observe_curriculum_metadata_delta_v1.py',
            ROOT/'tools/curriculum_confirmation_json_sources_v1.py',
            ROOT/'tools/prove_curriculum_placement_complete_data_origins_v1.py',
            ROOT/'runs/observe-complete-targeted-source-population-v2-private.py'):
        producer[str(path)] = sha(path)
    # Conflicts with earlier evidence can never be hidden by a later merge.
    bound = dict(evidence.files); merge_identities(bound, roots)
    merge_identities(bound, artifacts); merge_identities(bound, producer)
    out.mkdir(parents=True, exist_ok=False)
    write(out/'actual-artifact-role-catalogue-private.json', list(catalogue.values()))
    write(out/'actual-hash-edges-private.json', graph.edges)
    write(out/'new-metadata-observations-private.json', observations)
    result = dict(status=STATUS, candidate_identity=candidate['identity_sha256'],
        reference_identity=reference['identity_sha256'], seed_start=CRITERIA['seed_start'],
        seed_groups=CRITERIA['seed_groups'], reserved_source_data_hits=0, unreviewed_artifacts=[],
        preliminary_probe_used_as_final_proof=False, reserved_games_generated=0,
        training_lineages=lineages, producer_source_files=producer, artifact_files=artifacts,
        json_artifact_classifications={name:row for name,row in catalogue.items() if Path(name).suffix=='.json'},
        actual_metadata_observations=len(observations), actual_targeted_hash_edge_sources=edge_count,
        source_role_counts=dict(Counter(row['role'] for row in catalogue.values())),
        original_qualified_final_budget_and_independent_audit_required=True,
        old_closed_raw_values_not_recomputed=True, new_games=0, model_loaded=False,
        optimizer_updates=0, actual_final_seal_created=False, allocation=allocation,
        seconds=time.monotonic()-started)
    write(out/'source-proof-private.json', result)
    for path in out.glob('*-private.json'): bound[str(path)] = sha(path)
    write(out/'completion-binding-private.json', dict(files=bound,
        actual_final_seal_created=False, actual_final_confirmation_opened=False))
    print(json.dumps(dict(status=STATUS, artifacts=len(artifacts), new_metadata=len(observations),
        new_games=0, optimizer_updates=0, actual_final_confirmation_opened=False)), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out', type=Path, default=DEFAULT_OUT)
    prove(p.parse_args().out)
