"""Prepare the independently tested positive-evidence rule declared before training."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from incremental_score import prepare_positive_confirmation

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    result=prepare_positive_confirmation(args.source,args.out)
    print(json.dumps({'out':str(args.out),'checkpoint':result['checkpoint'],
                      'candidate_head_sha256':result['candidate_head_sha256'],
                      'gain_qualification':'positive paired95%CI lower bound;500 points desired only'}),flush=True)
