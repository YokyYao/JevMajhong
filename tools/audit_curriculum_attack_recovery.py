"""Audit the unchanged attack fit plus preregistered pre-calls recovery comparisons."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.curriculum_parent import read,rows,sha
from tools.curriculum_calls_recovery_parent import policy
from tools.attack_recovery_runtime import hooks,assessment
from tools import audit_curriculum_attack as numerical


def audit(registered,out,binding=None):
    reg=read(registered);p=policy();c=reg['curriculum']
    if c['run_id']!=p['experiment_run_id'] or reg.get('original_calls_quality_passed') is not False:
        raise ValueError('Wrong experimental recovery registration')
    with hooks(audit=numerical):numerical.audit(registered,out,binding)
    evidence=read(out)
    final=rows(ROOT/c['out']/'evaluations'/f"step-{reg['registered_final_step']:06d}"/'matches-private.jsonl')
    reference=ROOT/c['pre_calls_reference_evaluation_cache']
    for name in ('protocol.json','summary.json','matches-private.jsonl'):
        path=reference/name
        if reg['inputs'].get(str(path.resolve()))!=sha(path):raise ValueError('Pre-calls cache changed')
    comparison=numerical.paired_comparison(final,rows(reference/'matches-private.jsonl'))
    evidence['comparison_to_pre_calls_stage1']=comparison
    evidence['recovery_advancement']=assessment(evidence,comparison,p['recovery_targets'])
    evidence['experimental_transfer_not_original_quality_pass']=True
    evidence['pre_calls_reference_previously_audited_in_calls_evidence']=True
    out.write_text(json.dumps(evidence,indent=2)+'\n')
    print(json.dumps(dict(status='stage3_recovery_full_evidence_audited',recovery_advancement=evidence['recovery_advancement'])),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--registered',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--binding',type=Path)
    a=p.parse_args()
    try:audit(a.registered.resolve(),a.out.resolve(),a.binding.resolve() if a.binding else None)
    except BaseException as error:
        a.out.with_name(a.out.stem+'-error.json').write_text(json.dumps(dict(type=type(error).__name__,error=str(error)))+'\n')
        raise
