"""Seal only a completed six-source basic repair with every declared grade intact."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from tools.curriculum_basic_repair import MASS,grade
from tools.curriculum_parent import ParentNotReady,sha


def require_source_quality(reg,evidence,validation):
    if reg['curriculum'].get('source_specific_parent_quality_required') is not True:
        raise ParentNotReady('This seal requires the registered six-source basic repair')
    if (evidence.get('status')!='stage1_final_evidence_audit_passed'
        or evidence.get('source_specific_parent_seal_required') is not True
        or validation.get('local_step')!=reg['registered_final_step']):
        raise ParentNotReady('Completed full fit/source evidence required')
    counts=validation.get('source_stratum_counts',{})
    if set(counts)!=set(MASS) or any(type(n) is not int or n<=0 for n in counts.values()):
        raise ParentNotReady('All six actual validation source counts required')
    by=validation.get('riichi_by_source',{})
    if set(by)!={'legacy','teacher'}:raise ParentNotReady('Separate legacy and teacher confusion counts required')
    for source,c in by.items():
        if any(type(c.get(k,0)) is not int or c.get(k,0)<0 for k in ('tp','fn','tn','fp')):
            raise ParentNotReady('Invalid source confusion counts')
        if c.get('tp',0)+c.get('fn',0)!=counts[source+':riichi_yes'] or c.get('tn',0)+c.get('fp',0)!=counts[source+':riichi_no']:
            raise ParentNotReady('Source confusion counts differ from actual held-out windows')
    for kind in ('plain','riichi_yes','riichi_no'):
        if validation.get('stratum_counts',{}).get(kind)!=sum(counts[s+':'+kind] for s in by):
            raise ParentNotReady('Aggregate validation counts differ from source partitions')
    if any(validation.get('riichi_type_counts',{}).get(k,0)!=sum(c.get(k,0) for c in by.values()) for k in ('tp','fn','tn','fp')):
        raise ParentNotReady('Aggregate/source confusion counts disagree')
    calculated=grade(validation)
    if calculated!=evidence.get('basic_repair_source_quality') or calculated['original_aggregate_gate']!=evidence.get('stage1_advancement'):
        raise ParentNotReady('Stored grade differs from recomputed actual source metrics')
    if not calculated['all_source_quality_passed']:
        raise ParentNotReady('Aggregate, unchanged-legacy and teacher source grades must all pass')
    return calculated


def seal(registered,evidence,fit_exit,audit_exit,child_data_ready,out):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS') or socket.gethostname().startswith(('lo-','login')):
        raise ParentNotReady('Parent verification/sealing must run in an allocated GPU job')
    from tools.curriculum_parent import seal as original_seal
    read=lambda p:json.loads(Path(p).read_text())
    reg=read(registered);fit=ROOT/reg['curriculum']['out']
    validation=[json.loads(s) for s in (fit/'metrics/validation_metrics.jsonl').open()][-1]
    quality=require_source_quality(reg,read(evidence),validation)
    out=Path(out);intermediate=out.with_name(out.stem+'-aggregate-checked-private.json')
    if out.exists() or intermediate.exists():raise ParentNotReady('Never overwrite an existing parent seal or partial attempt')
    # Retain the original full-budget/reload/scheduler/replay checks. There is
    # no research override to turn a failed source grade into qualification.
    original_seal(registered,evidence,fit_exit,audit_exit,child_data_ready,intermediate)
    value=read(intermediate)
    value.update(source_specific_parent_quality_checked=True,basic_repair_source_quality=quality,
        source_specific_sealer='tools/seal_curriculum_basic_repair.py')
    for path in (Path(__file__),ROOT/'tools/curriculum_basic_repair.py'):
        value['bound_files'][str(path.resolve())]=sha(path)
    with out.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('registered','evidence','fit-exit','audit-exit','child-data-ready','out'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();seal(a.registered,a.evidence,a.fit_exit,a.audit_exit,a.child_data_ready,a.out)
