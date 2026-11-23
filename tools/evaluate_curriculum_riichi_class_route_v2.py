"""Execution-only repair: atomically update status, retain exclusive artifacts.

The bound v1 experiment and its failed partial directory remain unchanged.
Policy, frozen weights, threshold, baseline requirement and grid stay identical.
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import tempfile

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools import evaluate_curriculum_riichi_class_route_v1 as base
from tools.curriculum_transfer import sha

IMMUTABLE_WRITE=base.write


def status_write(path,value):
    if path.name!='status.json':return IMMUTABLE_WRITE(path,value)
    temporary=None
    try:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',dir=path.parent,prefix='.status-',suffix='.tmp',delete=False) as stream:
            temporary=Path(stream.name);json.dump(value,stream,indent=2);stream.write('\n')
        os.replace(temporary,path)
    finally:
        if temporary is not None and temporary.exists():temporary.unlink()


@contextmanager
def repaired_execution():
    old_context=base.context;old_write=base.write
    if old_write is not IMMUTABLE_WRITE:raise RuntimeError('No nested experiment patches')
    def context(config):
        result=list(old_context(config));files=result[5]
        for path in (Path(__file__),ROOT/'tests/core/test_curriculum_riichi_class_route_v2.py',
            ROOT/'docs/experiments/plans/curriculum-riichi-class-route-v2-20261123.md'):
            files[str(path.resolve())]=sha(path)
        return tuple(result)
    base.context=context;base.write=status_write
    try:yield
    finally:base.context=old_context;base.write=old_write


def run(config,phase,out=None):
    with repaired_execution():
        if phase=='preflight':return base.preflight(config,out)
        if phase=='evaluate':return base.evaluate(config)
        if phase=='audit':return base.audit(config,out)
        raise ValueError('Unknown phase')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--phase',choices=('preflight','evaluate','audit'),required=True);p.add_argument('--out',type=Path)
    a=p.parse_args()
    if a.phase!='evaluate' and a.out is None:p.error('--out is required for preflight/audit')
    run(a.config.resolve(),a.phase,a.out.resolve() if a.out else None)
