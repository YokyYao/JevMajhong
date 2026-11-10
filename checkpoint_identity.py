"""Validate upstream periodic checkpoints using their last-written trainer state."""
import json
from pathlib import Path

from call_policy import digest


def completed_checkpoint_files(directory, step):
    import torch
    path=Path(directory)
    files={p.name:digest(p) for p in path.iterdir() if p.is_file() and p.name!='trainer_state.pt'}
    required=('adapter_config.json','adapter_model.safetensors','head.pt','tokenizer.json',
              'tokenizer_config.json','chat_template.jinja','trainer_state.json')
    if any(n not in files for n in required):raise ValueError('Periodic checkpoint incomplete')
    header=json.loads((path/'trainer_state.json').read_text())
    state=torch.load(path/'trainer_state.pt',map_location='cpu',weights_only=True)
    if header['optimizer_step']!=step or state['optimizer_step']!=step:
        raise ValueError('Checkpoint step differs from completed trainer state')
    for name in ('adapter_model.safetensors','head.pt'):
        if state['checkpoint_sha256'].get(name)!=files[name]:
            raise ValueError('Checkpoint weights differ from completed trainer state')
    return files
