"""Extract public ordinary context pairs inside an allocated GPU job."""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys


def allocation_metadata(description,host):
    fields=dict(part.split('=',1) for part in description.split() if '=' in part)
    tres=dict(part.split('=',1) for part in fields.get('AllocTRES','').split(',') if '=' in part)
    gpu=any(key.startswith('gres/gpu') and value.isdigit() and int(value)>0 for key,value in tres.items())
    # These data jobs use one node, so exact matching also rejects execution
    # outside the parent's allocated node. No environment variable is forged.
    if fields.get('JobState')!='RUNNING' or fields.get('NodeList')!=host or not gpu:
        raise RuntimeError('Parent job does not grant a running GPU allocation on this node')
    return {key:fields[key] for key in ('JobId','JobState','NodeList','AllocTRES') if key in fields}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Data preparation requires an allocated GPU job, never login')
    description=subprocess.check_output(['scontrol','show','job',os.environ['SLURM_JOB_ID'],'--oneliner'],text=True)
    allocation=allocation_metadata(description,socket.gethostname())
    print(json.dumps({'hostname':socket.gethostname(),'job_id':os.environ['SLURM_JOB_ID'],
        'step_id':os.environ.get('SLURM_STEP_ID'),'allocated_step_cpus':os.environ.get('SLURM_CPUS_PER_TASK'),
        'parent_gpu_allocation':allocation,'work':'public data preparation on companion CPU, no model loading'}),flush=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from jevmahjong.endgame.native_context import prepare_contexts
    print(json.dumps(prepare_contexts(json.loads(args.config.read_text())),indent=2),flush=True)


if __name__=='__main__':main()
