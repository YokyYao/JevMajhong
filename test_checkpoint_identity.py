import json
from pathlib import Path
import tempfile
import unittest

import torch
from call_policy import digest
from checkpoint_identity import completed_checkpoint_files


class CheckpointIdentityTests(unittest.TestCase):
    def fixture(self,path):
        for name in ('adapter_config.json','adapter_model.safetensors','head.pt','tokenizer.json',
                     'tokenizer_config.json','chat_template.jinja'):
            (path/name).write_bytes(b'fixture')
        (path/'trainer_state.json').write_text(json.dumps({'optimizer_step':50}))
        torch.save({'optimizer_step':50,'checkpoint_sha256':{n:digest(path/n)
            for n in ('adapter_model.safetensors','head.pt')}},path/'trainer_state.pt')

    def test_actual_periodic_completion_format_needs_no_complete_json(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);self.fixture(path)
            files=completed_checkpoint_files(path,50)
            self.assertIn('head.pt',files)
            self.assertNotIn('trainer_state.pt',files)
            self.assertFalse((path/'complete.json').exists())

    def test_modified_weights_or_wrong_selected_step_are_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp);self.fixture(path)
            with self.assertRaises(ValueError):completed_checkpoint_files(path,100)
            (path/'adapter_model.safetensors').write_bytes(b'changed')
            with self.assertRaises(ValueError):completed_checkpoint_files(path,50)


if __name__=='__main__':unittest.main()
