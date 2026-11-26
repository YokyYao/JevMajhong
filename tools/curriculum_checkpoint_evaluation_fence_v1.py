"""Fence native evaluation until synchronous model AND optimizer saves finish."""
from pathlib import Path


def checkpoint_ready(root, step):
    directory=Path(root)/f'step-{step:06d}'
    names=('adapter_model.safetensors','head.pt')+(() if step==0 else ('trainer_state.pt',))
    return all((directory/name).is_file() for name in names)


class CheckpointEvaluationFence:
    def __init__(self, ready, evaluate):
        self.ready=ready;self.evaluate=evaluate;self.queued=set();self.finished=set()

    def validation(self, step):
        self.queued.add(step)
        # The wrapper writes the true pre-update snapshot before step0 validation.
        # Periodic validation occurs BEFORE upstream checkpoint save, so defer it.
        if step==0:self.flush()

    def flush(self):
        for step in sorted(self.queued-self.finished):
            if not self.ready(step):raise ValueError('Full saved checkpoint must precede native evaluation/update')
            self.evaluate(step);self.finished.add(step)
