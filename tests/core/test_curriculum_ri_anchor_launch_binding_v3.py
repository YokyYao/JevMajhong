import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tools import curriculum_ri_anchor_launch_binding_v3 as repair
from tools import curriculum_ri_anchor_fit_v1 as original
from tools.curriculum_calls_post_reach_data_v1 import write
from tools.curriculum_transfer import read, sha


class LaunchBindingTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name); self.run = self.root/'run'; self.control = self.root/'control'
        (self.run/'fit-plan').mkdir(parents=True); (self.control/'terminal-v1').mkdir(parents=True)
        self.entry = self.root/'original.slurm'; self.entry.write_text('original child entry\n')
        self.old_source = self.root/'old.py'; self.old_source.write_text('old source\n')
        self.new_source = self.root/'new.py'; self.new_source.write_text('new launch producer\n')
        self.extra = {str(self.new_source): sha(self.new_source)}
        self.registered = self.run/'fit-plan/registered.json'; self.queue = self.run/'fit-queue-private.json'
        self.prepared = self.run/'prepared.json'; self.preview = self.run/'preview.json'; self.binding = self.run/'activation.json'
        write(self.prepared, dict(files={str(self.old_source): sha(self.old_source)}))
        write(self.preview, dict(status='prospective', inputs={}))
        write(self.binding, dict(files={}, activation_job='900', current_capture_job='12'))
        (self.root/'logs').mkdir()
        write(self.root/'logs/curriculum-ri-balanced-replay-v2-exit-12.json', dict(phase='capture', exit_code=0))
        for path in (self.control/'stage3-fit/completion.json', self.control/'final-evidence-private.json',
                self.control/'audit-execution-private.json'):
            path.parent.mkdir(parents=True, exist_ok=True); write(path, {})
        self.captured = self.control/'terminal-v1/terminal-evidence-private.json'
        write(self.captured, dict(scheduler_proofs={role:dict(successful=True) for role in ('fit','audit')},
            strategy_grade=dict(passed=False)))
        write(self.control/'terminal-v1/completion-binding-private.json', dict(files={}))
        self.submissions = []; self.gate = self.run/'launch-gate-private.json'; self.launch = self.run/'launch-binding-private.json'
        def require_registration(reg):
            if reg['status'] != 'ri_anchor2596_registered_not_started':
                raise ValueError('Actual registration missing')
        def require_launch(path):
            if not self.gate.exists() or read(self.gate)['registered_sha256'] != sha(path):
                raise ValueError('Original gate absent or changed')
        def verify(files):
            if any(sha(path) != value for path, value in files.items()):
                raise ValueError('Source changed')
        contract = SimpleNamespace(CONTROL=self.control, require_registration=require_registration,
            require_launch=require_launch, verify_files=verify,
            completed_current=lambda *values: copy.deepcopy(values[-1]['strategy_grade']))
        def submit(role, dependency, limit, cpus=4, memory='48G'):
            self.assertTrue(self.registered.is_file()); self.assertTrue(self.gate.is_file())
            self.assertTrue(self.launch.is_file())
            self.assertEqual(read(self.launch)['registered_sha256'], sha(self.registered))
            self.submissions.append((role, dependency)); return str(1001+len(self.submissions)-1)
        def scheduler(job):
            dependencies={'1001':'afterok:900(unsatisfied)', '1002':'afterok:1001(unsatisfied)',
                '1003':'afterany:1001(unsatisfied),afterany:1002(unsatisfied)'}
            return SimpleNamespace(returncode=0,stdout=f'JobId={job} JobState=PENDING Command={self.entry} Dependency={dependencies[str(job)]}')
        changes=dict(ROOT=self.root,RUN=self.run,REGISTERED=self.registered,QUEUE=self.queue,
            PREPARED=self.prepared,PREVIEW=self.preview,BRIDGE=self.binding,ENTRY=self.entry,
            contract=contract,resource=lambda: 'finite GPU fixture only',verify_binding=lambda path: read(path),
            submit=submit,scheduler=scheduler,cancel_owned=lambda jobs: None)
        self.patch = patch.multiple(original, **changes); self.patch.start(); self.addCleanup(self.patch.stop)
        self.env = patch.dict(original.os.environ, {'SLURM_JOB_ID':'900'}); self.env.start(); self.addCleanup(self.env.stop)

    def run_activation(self):
        function = repair.activation_function(self.binding, self.extra)
        self.assertIs(function.__code__, original.activate.__code__)
        function()

    def test_complete_metadata_activation_creates_binding_before_fit_and_seals_queue(self):
        self.run_activation()
        self.assertEqual([role for role,_ in self.submissions], ['fit','audit','capture'])
        reg = read(self.registered); launch = read(self.launch); queue = read(self.queue)
        self.assertEqual(reg['inputs'][str(self.new_source)], sha(self.new_source))
        self.assertEqual(launch['files'][str(self.new_source)], sha(self.new_source))
        self.assertEqual(queue['files'][str(self.launch)], sha(self.launch))
        self.assertEqual(queue['files'][str(self.new_source)], sha(self.new_source))
        self.assertTrue(launch['launch_binding_written_before_fit_submission'])
        self.assertFalse(launch['numerical_recipe_changed'])
        self.assertEqual(self.submissions[0], ('fit','afterok:900'))

    def test_qualified_current_keeps_original_skip_and_creates_no_launch_or_child(self):
        value = read(self.captured); value['strategy_grade']['passed'] = True; self.captured.write_text(json.dumps(value))
        self.run_activation()
        self.assertEqual(self.submissions, []); self.assertFalse(self.launch.exists())
        self.assertFalse(self.registered.exists())
        self.assertEqual(read(self.run/'activation-outcome-private.json')['status'], 'skipped_qualified_current_follows_existing_3B')

    def test_infrastructure_failure_keeps_original_skip(self):
        value=read(self.captured); value['scheduler_proofs']['fit']['successful']=False;self.captured.write_text(json.dumps(value))
        self.run_activation()
        self.assertEqual(self.submissions, []); self.assertFalse(self.launch.exists())

    def test_existing_launch_binding_is_never_overwritten(self):
        self.launch.write_text('preserve existing launch\n')
        with self.assertRaisesRegex(ValueError,'Never overwrite'):
            self.run_activation()
        self.assertEqual(self.launch.read_text(), 'preserve existing launch\n')
        self.assertEqual(self.submissions, [])

    def test_missing_or_changed_gate_refused_before_launch_write(self):
        write(self.registered, dict(status='ri_anchor2596_registered_not_started',inputs={}))
        with self.assertRaisesRegex(ValueError,'gate'):
            repair.emit_launch_binding(self.registered,self.gate,self.launch,self.extra)
        write(self.gate,dict(registered_sha256='wrong'))
        with self.assertRaisesRegex(ValueError,'gate'):
            repair.emit_launch_binding(self.registered,self.gate,self.launch,self.extra)
        self.assertFalse(self.launch.exists())

    def test_missing_registration_or_changed_source_refused(self):
        with self.assertRaises(FileNotFoundError):
            repair.emit_launch_binding(self.registered,self.gate,self.launch,self.extra)
        write(self.registered,dict(status='ri_anchor2596_registered_not_started',inputs={}))
        write(self.gate,dict(registered_sha256=sha(self.registered)))
        self.new_source.write_text('changed')
        with self.assertRaisesRegex(ValueError,'Source changed'):
            repair.emit_launch_binding(self.registered,self.gate,self.launch,self.extra)
        self.assertFalse(self.launch.exists())

    def test_isolated_wrapper_does_not_replace_original_globals(self):
        before_write, before_submit, before_bridge = original.write, original.submit, original.BRIDGE
        function = repair.activation_function(self.binding,self.extra)
        self.assertIs(original.write,before_write); self.assertIs(original.submit,before_submit)
        self.assertEqual(original.BRIDGE,before_bridge)
        self.assertIsNot(function.__globals__,original.activate.__globals__)
        self.run_activation()
        self.assertIs(original.write,before_write);self.assertIs(original.submit,before_submit)


if __name__ == '__main__':
    unittest.main()
