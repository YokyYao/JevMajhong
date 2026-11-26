"""Full collection contracts and real-format recovery lineage at the v3 join."""
import ast
import copy
from pathlib import Path
import unittest
from unittest.mock import patch

from tools import curriculum_targeted_win_resumed_pipeline_v3 as pipeline


class CompleteWorldPipelineTests(unittest.TestCase):
    def fixture(self):
        folder = (pipeline.recovery.RUN/'shard-00/attempt-fixture').resolve()
        receipt_path = pipeline.ROOT/'logs/curriculum-targeted-win-resume-v3-exit-123.json'
        receipt = dict(job_id='123', exit_code=124)
        raw = ('JobId=123 JobState=TIMEOUT ExitCode=0:0 EndTime=2026-11-24T00:00:00 '
            f'Command={pipeline.recovery.ENTRY.resolve()}')
        proof = pipeline.terminal_record(raw, '123', pipeline.recovery.ENTRY)
        proof.update(receipt_path=str(receipt_path), exit_receipt=receipt)
        admission_path = folder.parent/'attempt-fixture-admission-private.json'
        manifest_path = folder/'world-cache-binding-private.json'
        complete = dict(binding_sha256='digest', world_cache_method=pipeline.METHOD,
            original_case_loop_unchanged=True, partial_worlds_admitted=False,
            exact_recovered_views=3, newly_processed_views=24, admission=str(admission_path),
            world_cache_manifest=str(manifest_path), world_cache_manifest_sha256='digest')
        admission = dict(status='actual_terminal_complete_cases_preserved_missing_cases_only',
            same_fixed_population_worlds_teacher_and_targets=True, completed_views=3,
            files={'original-fixed-case': 'digest'}, terminals=[proof])
        identity = dict(collection_sha256='digest', teacher_identity='teacher',
            producer_binding_sha256='digest', purpose='actual_fixed_label_collection', world_cache_method=pipeline.METHOD)
        manifest = dict(method=pipeline.METHOD, original_collection_identity=identity,
            partial_world_targets_admitted=False, all_current_arms_required=True,
            files={'immutable-complete-world': 'digest'})
        documents = {
            str(folder/'resumption-complete-private.json'): complete,
            str(admission_path): admission, str(manifest_path): manifest,
            str(receipt_path): receipt,
            str(pipeline.previous.RUN/'registered-private.json'): {'continuation_model': {'identity_sha256': 'teacher'}},
        }
        return folder, documents, complete, admission, manifest

    def check_fixture(self, fixture):
        folder, documents, *_ = fixture
        with patch.object(pipeline, 'read', side_effect=lambda p: documents[str(p)]), \
                patch.object(pipeline, 'sha', return_value='digest'):
            return pipeline.verify_recovery_lineage(folder)

    def test_full_pipeline_code_objects_and_original_namespace_are_preserved(self):
        for name in ('merge', 'prepare'):
            original = getattr(pipeline.previous, name)
            cloned = pipeline.pipeline_function(name)
            self.assertIs(cloned.__code__, original.__code__)
            self.assertIsNot(cloned.__globals__, original.__globals__)
            self.assertEqual(cloned.__globals__['ASSEMBLED'], pipeline.ASSEMBLED)
            self.assertNotEqual(original.__globals__['ASSEMBLED'], pipeline.ASSEMBLED)

    def test_source_calculations_roundtrip_after_only_registered_path_substitutions(self):
        for phase in ('merge', 'prepare'):
            _, name, path, tree = pipeline.original_source(phase)
            original = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
            restored = copy.deepcopy(tree.body[0]); substitutions = pipeline.SOURCE_SPECS[phase][2]
            for node in ast.walk(restored):
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in substitutions:
                    node.value = ast.parse(substitutions[node.targets[0].id][0], mode='eval').body
            self.assertEqual(ast.dump(original), ast.dump(restored))

    def test_partial_or_duplicate_shards_cannot_execute_original_merge(self):
        with patch.object(pipeline.previous, 'resource', return_value='fixture'), \
                patch.object(pipeline, 'verify_entry', return_value=({'files': {}}, {})):
            merge = pipeline.pipeline_function('merge')
            for folders in ([], [Path('/same')]*4):
                with self.assertRaisesRegex(ValueError, 'four distinct'): merge(folders, [None]*4)
        with self.assertRaisesRegex(ValueError, 'full27-view'):
            pipeline.previous.verify_complete(Path('/absent-fixture'), 0, {})

    def test_complete_v3_lineage_binds_original_cases_and_world_cache(self):
        files = self.check_fixture(self.fixture())
        self.assertEqual(files['immutable-complete-world'], 'digest')
        self.assertEqual(files['original-fixed-case'], 'digest')
        self.assertTrue(any(p.endswith('world-cache-binding-private.json') for p in files))

    def test_wrong_manifest_identity_path_partial_target_or_hash_is_rejected(self):
        for mutation in ('test_identity', 'manifest_path', 'partial', 'manifest_hash', 'world_hash', 'count'):
            fixture = self.fixture(); _, _, complete, _, manifest = fixture
            if mutation == 'test_identity': manifest['original_collection_identity']['purpose'] = 'software_roundtrip_only_not_production_reusable'
            elif mutation == 'manifest_path': complete['world_cache_manifest'] = '/another-manifest'
            elif mutation == 'partial': manifest['partial_world_targets_admitted'] = True
            elif mutation == 'manifest_hash': complete['world_cache_manifest_sha256'] = 'changed'
            elif mutation == 'world_hash': manifest['files']['immutable-complete-world'] = 'changed'
            else: complete['newly_processed_views'] = 23
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.check_fixture(fixture)

    def test_live_unknown_or_changed_prior_writer_is_rejected(self):
        for mutation in ('live', 'unknown', 'receipt_path', 'terminal'):
            fixture = self.fixture(); proof = fixture[3]['terminals'][0]
            if mutation == 'live': proof['raw'] = proof['raw'].replace('JobState=TIMEOUT', 'JobState=RUNNING')
            elif mutation == 'unknown': proof['raw'] = proof['raw'].replace(str(pipeline.recovery.ENTRY.resolve()), '/unregistered-producer')
            elif mutation == 'receipt_path': proof['receipt_path'] = '/wrong-receipt'
            else: proof['state'] = 'COMPLETED'
            with self.subTest(mutation=mutation), self.assertRaises(ValueError): self.check_fixture(fixture)

    def test_v2_lineage_uses_its_original_frozen_validator(self):
        expected = {'old-admission': 'old-digest'}
        with patch.object(pipeline.previous, 'verify_recovery_lineage', return_value=expected) as validator:
            folder = pipeline.previous.RESUME/'shard-00/attempt-001'
            self.assertIs(pipeline.verify_recovery_lineage(folder), expected)
            validator.assert_called_once_with(folder.resolve())


if __name__ == '__main__': unittest.main()
