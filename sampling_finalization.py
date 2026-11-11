"""Normalize SDK checkpoint paths before exporting repository-relative results."""
from pathlib import Path


def canonical_checkpoint(root, run, selected):
    root = Path(root).resolve()
    run = Path(run)
    run = (root / run).resolve() if not run.is_absolute() else run.resolve()
    run.relative_to(root)
    selected = Path(selected)
    if selected.is_absolute():
        raise ValueError('SDK selected checkpoint must be relative to its run')
    checkpoint = (run / selected).resolve()
    checkpoint.relative_to(run)
    return str(checkpoint.relative_to(root))
