import json
from pathlib import Path
import tempfile
import unittest
from tools import evaluate_curriculum_riichi_class_route_v1 as base
from tools.evaluate_curriculum_riichi_class_route_v2 import status_write,repaired_execution


class RouteStatusRepair(unittest.TestCase):
    def test_actual_failure_first_and_subsequent_status_writes(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'status.json'
            status_write(path,{'complete':1});status_write(path,{'complete':2});status_write(path,{'complete':16})
            self.assertEqual(json.loads(path.read_text()),{'complete':16})
            self.assertEqual(list(Path(folder).iterdir()),[path])

    def test_other_artifacts_still_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'summary.json';status_write(path,{'value':1})
            with self.assertRaises(FileExistsError):status_write(path,{'value':2})
            self.assertEqual(json.loads(path.read_text()),{'value':1})

    def test_dispatch_is_restored_on_failure(self):
        old_write,old_context=base.write,base.context
        with self.assertRaisesRegex(RuntimeError,'fixture'):
            with repaired_execution():
                self.assertIs(base.write,status_write)
                self.assertIsNot(base.context,old_context)
                raise RuntimeError('fixture')
        self.assertIs(base.write,old_write);self.assertIs(base.context,old_context)
