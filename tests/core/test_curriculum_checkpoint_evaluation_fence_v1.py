import tempfile
from pathlib import Path
import unittest
from tools.curriculum_checkpoint_evaluation_fence_v1 import CheckpointEvaluationFence,checkpoint_ready


class EvaluationSaveOrderingTests(unittest.TestCase):
    def test_validation_before_save_waits_and_next_batch_sees_durable_optimizer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);events=[]
            fence=CheckpointEvaluationFence(lambda s:checkpoint_ready(root,s),lambda s:events.append(('native',s)))
            fence.validation(600)
            self.assertEqual(events,[])
            p=root/'step-000600';p.mkdir()
            for name in ('adapter_model.safetensors','head.pt'):(p/name).write_bytes(b'fixture')
            with self.assertRaises(ValueError):fence.flush()
            self.assertEqual(events,[])
            (p/'trainer_state.pt').write_bytes(b'fixture')
            fence.flush();events.append(('next_batch',601));fence.flush()
            self.assertEqual(events,[('native',600),('next_batch',601)])

    def test_final_checkpoint_is_flushed_even_without_another_batch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);events=[]
            p=root/'step-000000';p.mkdir()
            for name in ('adapter_model.safetensors','head.pt'):(p/name).write_bytes(b'fixture')
            fence=CheckpointEvaluationFence(lambda s:checkpoint_ready(root,s),events.append)
            fence.validation(0)
            fence.validation(2400)
            self.assertEqual(events,[0])
            p=root/'step-002400';p.mkdir()
            for name in ('adapter_model.safetensors','head.pt','trainer_state.pt'):(p/name).write_bytes(b'fixture')
            fence.flush()
            self.assertEqual(events,[0,2400])
