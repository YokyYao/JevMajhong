"""Extract public ordinary context pairs inside an allocated GPU job."""
import argparse
import json
import os
from pathlib import Path
import socket
import sys


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True);args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURM_JOB_GPUS') or socket.gethostname().startswith(('lo-','login')):
        raise RuntimeError('Data preparation requires an allocated GPU job, never login')
    print(json.dumps({'hostname':socket.gethostname(),'job_id':os.environ['SLURM_JOB_ID'],
        'step_id':os.environ.get('SLURM_STEP_ID'),'allocated_step_cpus':os.environ.get('SLURM_CPUS_PER_TASK'),
        'work':'public data preparation on companion CPU, no model loading'}),flush=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
    from jevmahjong.endgame.native_context import prepare_contexts
    print(json.dumps(prepare_contexts(json.loads(args.config.read_text())),indent=2),flush=True)


if __name__=='__main__':main()
