"""Capture every pure-call opportunity, including states after earlier calls."""
import argparse
import json
from pathlib import Path
from jevmahjong.data.native_states import CALL_METHOD
from .collector import NativeCollector
from .defense_states import collect_native_inputs


class CallStateCollector(NativeCollector):
    def __init__(self,config,device='cuda'):
        if config.get('behavior_heads'):raise ValueError('Original frozen Play collection required')
        super().__init__(config,device)

    def act(self,obs,frozen=False,stochastic=False,trace=None):
        captured=[] if trace is not None else None
        chosen=super().act(obs,frozen,stochastic,captured)
        if captured:trace.extend(row for row in captured if row['scope']=='calling')
        return chosen


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--resume',action='store_true');parser.add_argument('--max-walls',type=int)
    args=parser.parse_args()
    collect_native_inputs(json.loads(args.config.read_text()),args.resume,args.max_walls,
        collector=CallStateCollector,method=CALL_METHOD,scope='calling')


if __name__=='__main__':main()
