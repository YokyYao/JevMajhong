"""A recovered collection must satisfy the original full merge/projection."""
import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import curriculum_targeted_win_resumed_pipeline_v2 as pipeline


class ResumedTargetedPipelineTests(unittest.TestCase):
    def test_merge_changes_only_its_root_path(self):
        _, name, path, adapted = pipeline.original_source('merge')
        original = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
        restored = copy.deepcopy(adapted.body[0])
        assignment = next(n for n in ast.walk(restored) if isinstance(n, ast.Assign)
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id == 'root')
        assignment.value = ast.parse(pipeline.SOURCE_SPECS['merge'][2]['root'][0], mode='eval').body
        self.assertEqual(ast.dump(original), ast.dump(restored))

    def test_projection_changes_only_control_and_output_paths(self):
        _, name, path, adapted = pipeline.original_source('prepare')
        original = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == name)
        restored = copy.deepcopy(adapted.body[0]); replacements = pipeline.SOURCE_SPECS['prepare'][2]
        for node in ast.walk(restored):
            if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in replacements:
                node.value = ast.parse(replacements[node.targets[0].id][0], mode='eval').body
        self.assertEqual(ast.dump(original), ast.dump(restored))

    def test_changed_or_absent_original_path_is_rejected(self):
        source = 'def merge():\n    root = ROOT/"unexpected"\n'
        with self.assertRaises(ValueError): pipeline.adapted_function(source, 'merge', pipeline.SOURCE_SPECS['merge'][2])
        with self.assertRaises(ValueError): pipeline.adapted_function('def merge():\n    pass\n', 'merge', pipeline.SOURCE_SPECS['merge'][2])

    def test_full_fixture_requires_exact_population_teacher_and_counts(self):
        selected = [dict(id=f'fixture-{i}') for i in range(108)]
        reg = dict(selected=selected, continuation_model={'identity_sha256': 'teacher'})
        launch = dict(selected=selected[:27], parent_registration_sha256='hash',
            model=reg['continuation_model'], job_id='123', node='ws-fixture')
        summary = dict(status='fixed_targeted_win_label_shard_complete_candidate_targets_not_fit_admitted',
            selected_views=27, teacher_model='teacher', job_id='123', node='ws-fixture',
            counts=dict(labelled_views=25, sampling_quarantine=2), files={})
        with tempfile.TemporaryDirectory() as directory, patch.object(pipeline, 'sha', return_value='hash'):
            folder = Path(directory)
            (folder/'registered-private.json').write_text(json.dumps(launch))
            path = folder/'summary-private.json'; path.write_text(json.dumps(summary))
            self.assertEqual(len(pipeline.verify_complete(folder, 0, reg)), 2)
            for field, value in [('selected_views', 26), ('teacher_model', 'other'),
                    ('counts', {'labelled_views': 24, 'sampling_quarantine': 2})]:
                changed = dict(summary, **{field: value}); path.write_text(json.dumps(changed))
                with self.subTest(field=field), self.assertRaises(ValueError): pipeline.verify_complete(folder, 0, reg)

    def test_partial_folder_is_not_merge_ready(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, 'full27-view'): pipeline.verify_complete(Path(directory), 0, {})

    def test_recovery_lineage_rejects_wrong_binding_and_live_ancestor(self):
        complete = dict(binding_sha256='hash', original_case_loop_unchanged=True,
            partial_worlds_admitted=False, exact_recovered_views=2, newly_processed_views=25, admission='fixture-admission')
        receipt = dict(job_id='123', phase='label', exit_code=0)
        raw = f'JobId=123 JobState=RUNNING EndTime=Unknown Command={pipeline.ROOT}/cluster/curriculum-targeted-win-labels-v1.slurm'
        admission = dict(status='actual_terminal_complete_cases_preserved_missing_cases_only',
            same_fixed_population_worlds_teacher_and_targets=True, completed_views=2, files={},
            terminals=[dict(raw=raw, job_id='123', exit_receipt=receipt, receipt_path='fixture-receipt')])
        def reader(path):
            if Path(path).name == 'resumption-complete-private.json': return complete
            return admission if str(path) == 'fixture-admission' else receipt
        with patch.object(pipeline, 'read', side_effect=reader), patch.object(pipeline, 'sha', return_value='hash'):
            with self.assertRaises(ValueError): pipeline.verify_recovery_lineage(Path('/fixture'))
            complete['binding_sha256'] = 'wrong'
            with self.assertRaises(ValueError): pipeline.verify_recovery_lineage(Path('/fixture'))


if __name__ == '__main__': unittest.main()
