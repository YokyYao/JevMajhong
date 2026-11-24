"""Recovered exposure preparation keeps the original full-data calculations."""
import ast
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tools import prepare_curriculum_recovered_exposures_v1 as prepare


class RecoveredExposuresTests(unittest.TestCase):
    def test_both_original_calculations_roundtrip_after_only_path_substitution(self):
        for phase in ('mix', 'exposures'):
            _, path, tree = prepare.original_source(phase)
            original = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name == 'prepare')
            restored = copy.deepcopy(tree.body[0]); replacements = prepare.SOURCES[phase][1]
            for node in ast.walk(restored):
                if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name) and node.targets[0].id in replacements:
                    node.value = ast.parse(replacements[node.targets[0].id][0], mode='eval').body
            self.assertEqual(ast.dump(original), ast.dump(restored))

    def test_changed_or_missing_original_path_cannot_be_silently_adapted(self):
        replacements = prepare.SOURCES['mix'][1]
        for source in ('def prepare():\n    parent_path = ROOT/"other"\n    out = ROOT/"other"\n', 'def prepare():\n    pass\n'):
            with self.assertRaises(ValueError): prepare.adapted_function(source, 'prepare', replacements)

    def test_producer_binding_join_is_exact_and_does_not_mutate_either_input(self):
        original = {'same': 'one', 'original': 'two'}; extra = {'same': 'one', 'recovery': 'three'}
        before = copy.deepcopy((original, extra)); combined = prepare.combine_files(original, extra)
        self.assertEqual(combined, {'same': 'one', 'original': 'two', 'recovery': 'three'})
        self.assertEqual((original, extra), before)
        with self.assertRaises(ValueError): prepare.combine_files(original, {'same': 'changed'})

    def test_exposure_config_changes_only_source_and_data_locations(self):
        new = prepare.read(prepare.CONFIG)
        old = prepare.read(prepare.ROOT/'configs/training/curriculum-placement-exposures-v1.json')
        locations = {'source_registration', 'data_ready', 'mix_proposal', 'out'}
        self.assertEqual({k:v for k,v in new.items() if k not in locations}, {k:v for k,v in old.items() if k not in locations})
        self.assertEqual((prepare.ROOT/new['data_ready']).resolve(), prepare.DATA.resolve())
        self.assertEqual((prepare.ROOT/new['out']).resolve(), prepare.EXPOSURES.resolve())

    def test_complete_projection_and_unchanged_producer_hashes_are_required(self):
        for mutation in ('none', 'partial', 'changed', 'wrong_status', 'already_fit'):
            producer = dict(actual_full_original_projection_complete=True, files={'fixed-input': 'digest'})
            ready = dict(status='placement_all_sources_including_targeted_win_special_prepared_not_fit', optimizer_updates=0)
            if mutation == 'partial': producer['actual_full_original_projection_complete'] = False
            elif mutation == 'changed': producer['files']['fixed-input'] = 'changed'
            elif mutation == 'wrong_status': ready['status'] = 'partial'
            elif mutation == 'already_fit': ready['optimizer_updates'] = 1
            def reader(path):
                return producer if Path(path).name == 'resumed-producer-binding-private.json' else ready
            with self.subTest(mutation=mutation), patch.object(prepare, 'read', side_effect=reader), \
                    patch.object(prepare, 'sha', return_value='digest'):
                if mutation == 'none': self.assertIn('fixed-input', prepare.completed_input())
                else:
                    with self.assertRaises(ValueError): prepare.completed_input()

    def test_missing_full_input_or_existing_output_cannot_execute_original(self):
        binding = dict(status='recovered_exposure_entry_bound_not_executed', files={})
        with patch.object(prepare, 'resource', return_value='fixture'), patch.object(prepare, 'read', return_value=binding), \
                patch.object(prepare, 'completed_input', side_effect=FileNotFoundError('actual full projection missing')), \
                patch.object(prepare, 'original_source') as original:
            with self.assertRaises(FileNotFoundError): prepare.execute('mix')
            original.assert_not_called()
        with tempfile.TemporaryDirectory() as directory, patch.object(prepare, 'resource', return_value='fixture'), \
                patch.object(prepare, 'read', return_value=binding), patch.object(prepare, 'completed_input', return_value={}), \
                patch.object(prepare, 'sha', return_value='digest'), patch.object(prepare, 'MIX', Path(directory)), \
                patch.object(prepare, 'original_source') as original:
            with self.assertRaisesRegex(ValueError, 'overwrite'): prepare.execute('mix')
            original.assert_not_called()


if __name__ == '__main__': unittest.main()
