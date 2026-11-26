"""Recovery changes scheduling and I/O, never the bound case calculations."""
import ast
from collections import Counter
import copy
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import curriculum_targeted_win_resume_v2 as resume


class TargetedWinResumeTests(unittest.TestCase):
    def test_original_case_calculations_remain_exact_AST(self):
        source = resume.COLLECTOR.read_text()
        _, before = resume.original_loop(ast.parse(source))
        _, after = resume.original_loop(resume.resumed_function_tree(source))
        self.assertEqual(ast.dump(ast.Module(body=before.body, type_ignores=[])),
            ast.dump(ast.Module(body=after.body[1:], type_ignores=[])))
        self.assertIsInstance(after.body[0], ast.If)

    def test_changed_output_or_missing_loop_is_rejected(self):
        source = resume.COLLECTOR.read_text()
        with self.assertRaises(ValueError):
            resume.resumed_function_tree(source.replace("root/'shards'", "root/'other'"))
        with self.assertRaises(ValueError):
            resume.resumed_function_tree(source.replace('enumerate(specs)', 'enumerate(other)'))

    def test_conflicting_results_cannot_be_selected(self):
        a = {'x': dict(kind='case', case={'value': 1})}
        resume.combine(a, {'x': dict(kind='case', case={'value': 1})})
        for row in (dict(kind='case', case={'value': 2}), dict(kind='quarantine', row={})):
            with self.assertRaises(ValueError): resume.combine(a, {'x': row})
        self.assertEqual(a['x']['case'], {'value': 1})

    def test_explicit_quarantine_is_reused_without_partial_label(self):
        labels = io.StringIO(); quarantine = io.StringIO(); counts = Counter()
        original = dict(source_id='x', reason='public_prior_sampling_exhausted')
        resume.reuse(dict(kind='quarantine', row=original), {}, 0, None,
            labels, quarantine, counts, {})
        self.assertFalse(labels.getvalue()); self.assertIn('public_prior_sampling_exhausted', quarantine.getvalue())
        self.assertEqual(counts, {'sampling_quarantine': 1})

    def test_complete_case_reuse_preserves_source_file(self):
        case = dict(candidate_label={'same': 'label'}, soft_label=dict(label='pon', point_choice='pon',
            independent_block_choices_agree=True, every_leave_one_world_choice_agrees=False),
            blocks=[dict(baseline='pass', pairs=[{'pass': {}, 'pon': {}}])], planner_counts={'calls': 3})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root/'original.json'; source.write_text('original bytes')
            dest = root/'reused'; dest.mkdir(); counts = Counter(); contexts = {}; labels = io.StringIO()
            value = dict(kind='case', case=case, path=str(source), sha256='correct')
            with patch.object(resume, 'sha', return_value='correct'):
                resume.reuse(value, dict(split='train', bucket='test'), 4, dest,
                    labels, io.StringIO(), counts, contexts)
            self.assertEqual(source.read_text(), 'original bytes')
            self.assertEqual((dest/'case-004.json').resolve(), source)
            self.assertEqual(counts['complete_outer_arms'], 2)
            self.assertEqual(counts['labelled_views'], 1)
            self.assertEqual(counts['future:calls'], 3)

    def test_live_prior_writer_is_rejected(self):
        folder = resume.RUN/'shards/shard-00'
        raw = f'JobId=123 JobState=RUNNING NodeList=ws-test Command={resume.ROOT}/cluster/curriculum-targeted-win-labels-v1.slurm EndTime=Unknown'
        def reader(path):
            return dict(job_id='123', node='ws-test') if Path(path).name == 'registered-private.json' else dict(job_id='123', phase='label', exit_code=0)
        with patch.object(resume, 'read', side_effect=reader), patch.object(resume, 'scheduler',
                return_value=SimpleNamespace(returncode=0, stdout=raw)):
            with self.assertRaises(ValueError): resume.require_terminal(folder)

    def test_missing_terminal_cannot_be_fabricated(self):
        folder = resume.RUN/'shards/shard-00'
        def reader(path):
            return dict(job_id='123', node='ws-test') if Path(path).name == 'registered-private.json' else dict(job_id='123', phase='label', exit_code=0)
        with patch.object(resume, 'read', side_effect=reader), patch.object(resume, 'scheduler',
                return_value=SimpleNamespace(returncode=1, stdout='')):
            with self.assertRaises(ValueError): resume.require_terminal(folder)


if __name__ == '__main__': unittest.main()
