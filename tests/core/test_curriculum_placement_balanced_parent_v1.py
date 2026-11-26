"""Alternate placement parent cannot borrow the original route's evidence."""
import ast
import copy
import inspect
from pathlib import Path
from types import SimpleNamespace
import unittest

from tools import curriculum_placement_balanced_parent_v1 as alternate
from tools import seal_curriculum_placement_retained_parent_v1 as original
from tools import curriculum_placement_retained_parent_v1 as retained
from tests.core.test_curriculum_placement_retained_seal_v1 import phases, data_fixture


def balanced_phases():
    cap, queue, captured, receipt, receipts = phases()
    updated = {}
    for path, value in receipts.items():
        updated[path.with_name(path.name.replace('curriculum-yaku-parent-exit-', alternate.PREFIX))] = value
    for proof in cap['scheduler_proofs'].values():
        proof['raw'] = proof['raw'].replace(str(retained.ENTRY.resolve()), str(alternate.ENTRY.resolve()))
    captured['raw'] = captured['raw'].replace(str(retained.ENTRY.resolve()), str(alternate.ENTRY.resolve()))
    return cap, queue, captured, receipt, updated


class BalancedPlacementParentTests(unittest.TestCase):
    def test_exact_four_balanced_producer_exits_required(self):
        cap, queue, captured, receipt, receipts = balanced_phases()
        exits, paths = alternate.phase_function()(cap, queue, captured, receipt, receipts.__getitem__)
        self.assertEqual(set(exits), {'prepare', 'fit', 'audit', 'capture'})
        self.assertTrue(all(path.name.startswith(alternate.PREFIX) for path in paths))

    def test_original_producer_or_failed_capture_cannot_substitute(self):
        with self.assertRaises(ValueError):
            cap, queue, captured, receipt, receipts = phases()
            alternate.phase_function()(cap, queue, captured, receipt, receipts.__getitem__)
        cap, queue, captured, receipt, receipts = balanced_phases()
        captured['raw'] = captured['raw'].replace('JobState=COMPLETED', 'JobState=FAILED')
        with self.assertRaises(ValueError):
            alternate.phase_function()(cap, queue, captured, receipt, receipts.__getitem__)

    def test_initializer_offset_and_parent_identity_are_explicit(self):
        parent = dict(curriculum=dict(run_id='jev-curriculum-yaku-balanced-parent-v1',
            stage='stage3B-yaku-balanced-parent-v1', stage_offset=3468,
            out='runs/curriculum-yaku-balanced-parent-v1/stage3B-fit'),
            training_config=dict(parent_initialization=dict(method=alternate.INITIALIZER, selected_step=2472)))
        alternate.require_lineage(parent)
        for key, value in [('stage_offset', 3396), ('run_id', 'original-retained-route')]:
            bad = copy.deepcopy(parent); bad['curriculum'][key] = value
            with self.assertRaises(ValueError): alternate.require_lineage(bad)
        bad = copy.deepcopy(parent); bad['training_config']['parent_initialization']['selected_step'] = 2400
        with self.assertRaises(ValueError): alternate.require_lineage(bad)

    def test_no_implicit_or_original_parent_seal(self):
        sealed = dict(placement_parent_lineage_method=alternate.METHOD,
            balanced_parent_selected_step=2472, placement_parent_selected_step=2500,
            parent_registration=str(alternate.PARENT))
        alternate.require_sealed_lineage(sealed)
        for key in sealed:
            bad = dict(sealed); del bad[key]
            with self.assertRaises((KeyError, ValueError)): alternate.require_sealed_lineage(bad)
        with self.assertRaises(ValueError):
            alternate.require_sealed_lineage(dict(sealed, parent_registration=str(original.PARENT)))

    def test_source_transform_changes_only_declared_path_constants(self):
        substitutions = [
            (original.phase_proofs, {'curriculum-yaku-parent-exit-': (alternate.PREFIX, 2)}),
            (original.parent_artifacts, {
                'yaku-parent-terminal-v1/terminal-evidence-private.json': (alternate.CAP, 1),
                'curriculum-yaku-parent-exit-': (alternate.PREFIX, 1)}),
            (retained.verify_parent, {
                'yaku-parent-terminal-v1/terminal-evidence-private.json': (alternate.CAP, 1),
                'curriculum-progression-parent-retention-v1/yaku-parent-queue-v1-private.json':
                    ('curriculum-ri-balanced-replay-fit-v1/balanced-yaku-queue-v1-private.json', 1),
                'curriculum-yaku-parent-exit-': (alternate.PREFIX, 2)}),
            (original.seal, {}),
        ]
        for function, replacements in substitutions:
            actual = alternate.adapted_tree(function, replacements)
            reverse = {new: old for old, (new, _) in replacements.items()}
            for node in ast.walk(actual):
                if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in reverse:
                    node.value = reverse[node.value]
            self.assertEqual(ast.dump(actual), ast.dump(ast.parse(inspect.getsource(function))))
        with self.assertRaises(ValueError):
            alternate.adapted_tree(original.phase_proofs, {'nonexistent-path': ('replacement', 1)})

    def test_original_full57_dataset_requirements_still_reject_incomplete_pools(self):
        source = dict(status='placement_all_sources_including_targeted_win_special_prepared_not_fit')
        original.require_data(data_fixture(), source)
        bad = data_fixture(); bad['source_unique_records'].pop(next(iter(bad['source_unique_records'])))
        with self.assertRaises(ValueError): original.require_data(bad, source)

    def test_parent_loader_restores_global_after_failure(self):
        initialization = SimpleNamespace(verified_parent=object())
        before = initialization.verified_parent
        with self.assertRaises(RuntimeError):
            with alternate.parent_loader(initialization):
                self.assertIs(initialization.verified_parent, alternate.verify_parent)
                raise RuntimeError('fixture-only')
        self.assertIs(initialization.verified_parent, before)


if __name__ == '__main__': unittest.main()
