"""Pinned Mortal produces source play, never counterfactual terminal labels.

Fresh native source games have consistent real private histories. This adapter
does NOT admit a Bot initialized with an original history after hand sampling.
External Mortal source and weights are dependencies, not redistributed here.
"""
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import socket
import sys

from tools.collect_curriculum_endgame_cases import sha

WEIGHT_SHA256='bfb3a6c072aa0bfd4171a9cdc77cb6c02ae42cde920843f9e5784394f23447d8'
BINARY_SHA256='3aeb09112198f6babe9bba4dd392ea009669ee7ba5d33cd3cd4242c737bb004b'


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    result=importlib.util.module_from_spec(spec);sys.modules[name]=result;spec.loader.exec_module(result)
    return result


def public_events(events,seat):
    result=[]
    for raw in events:
        event=json.loads(raw) if isinstance(raw,str) else dict(raw)
        if event['type']=='start_game':event.update(id=seat,names=['0','1','2','3'])
        if event['type']=='start_kyoku' and any(t!='?' for p,hand in enumerate(event['tehais']) if p!=seat for t in hand):
            raise ValueError('An opponent initial hand entered the source actor')
        if event['type']=='tsumo' and event['actor']!=seat and event['pai']!='?':
            raise ValueError('An opponent draw entered the source actor')
        result.append(event)
    return result


class MortalSource:
    def __init__(self,root):
        assert os.environ.get('SLURM_JOB_ID') and os.environ.get('SLURM_JOB_GPUS')
        assert not socket.gethostname().startswith(('lo-','login'))
        import torch
        root=Path(root)
        binary=root/'mortal-adapter-reference/libriichi/libriichi-3.12-x86_64-unknown-linux-gnu.so'
        weights=root/'mortal-298k.pth'
        if sha(weights)!=WEIGHT_SHA256 or sha(binary)!=BINARY_SHA256:raise ValueError('Unregistered Mortal artifacts')
        paths=[weights,binary,root/'mortal-adapter-reference/_libriichi_loader.py',
            root/'mortal-upstream/mortal/model.py',root/'mortal-upstream/mortal/engine.py',Path(__file__)]
        self.bindings={str(path.resolve()):sha(path) for path in paths}
        loader=module('_curriculum_mortal_loader',paths[2]);loader.load()
        import libriichi
        if Path(libriichi.__file__).resolve()!=binary.resolve():raise ValueError('Another libriichi binary is active')
        model=module('_curriculum_mortal_model',paths[3]);engine=module('_curriculum_mortal_engine',paths[4])
        saved=torch.load(weights,map_location='cpu',weights_only=True);cfg=saved['config']
        if cfg['control']['version']!=4 or cfg['resnet']['conv_channels']!=192 or cfg['resnet']['num_blocks']!=40:
            raise ValueError('Mortal architecture mismatch')
        brain=model.Brain(version=4,conv_channels=192,num_blocks=40);dqn=model.DQN(version=4)
        brain.load_state_dict(saved['mortal'],strict=True);dqn.load_state_dict(saved['current_dqn'],strict=True)
        self.engine=engine.MortalEngine(brain,dqn,False,4,device=torch.device('cuda'),enable_amp=False,
            enable_rule_based_agari_guard=False,name='Mortal-298k-source-only')
        from libriichi.mjai import Bot
        from libriichi.state import PlayerState
        self.Bot=Bot;self.PlayerState=PlayerState;self.counts=Counter()

    def player(self,seat):return MortalSourceSeat(self,seat)


class MortalSourceSeat:
    def __init__(self,source,seat):
        self.source=source;self.seat=seat;self.bot=source.Bot(source.engine,seat);self.mirror=source.PlayerState(seat)

    def act(self,obs):
        from riichienv import ActionType
        from jevmahjong.core.state import legal_candidates,signature
        if obs.player_id!=self.seat:raise ValueError('Wrong source actor')
        events=public_events(obs.events,self.seat);actions=legal_candidates(obs)
        wins=[a for a in actions if a.action_type in (ActionType.RON,ActionType.TSUMO)]
        response=None
        for i,event in enumerate(events):
            encoded=json.dumps(event)
            self.mirror.update(encoded)
            response=self.bot.react(encoded,can_act=i==len(events)-1 and len(actions)>1 and not wins)
        if list(self.mirror.tehai)!=[sum(t//4==i for t in obs.hand) for i in range(34)]:
            raise ValueError('Source teacher event history disagrees with current own hand')
        self.source.counts['own_hand_history_checks']+=1
        if len(actions)==1:chosen=actions[0]
        elif wins:chosen=wins[0];self.source.counts['forced_legal_wins']+=1
        else:
            if response is None:raise ValueError('Mortal returned no action for an actual native choice')
            decision=json.loads(response);decision.pop('meta',None)
            proposed=obs.select_action_from_mjai(decision)
            if proposed is None:raise ValueError('Mortal action could not map to the native menu')
            matches=[a for a in actions if signature(a)==signature(proposed)]
            if len(matches)!=1:raise ValueError('Mortal action has no unique native identity')
            chosen=matches[0];self.source.counts['network_choices']+=1
        self.source.counts['selected_'+json.loads(chosen.to_mjai())['type']]+=1
        return chosen
