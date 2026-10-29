import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from jevany.model import PointerHead
from head_rl import PPO, development_gate, mixed_probabilities, reference_advantages, summarize


class ReferenceBaselineTests(unittest.TestCase):
    def setUp(self):
        self.config = json.loads(Path('head-rl-score-reference.json').read_text())

    def fixtures(self):
        common = {'seed':1,'seat':0,'engine_seed':1<<32,'game_mode':'4p-red-east','old_opponent':False}
        controls = [dict(common,replica=-1,reward=1.,trainable_decisions=0,frozen_control=True)]
        games = [dict(common,replica=i,reward=r,trainable_decisions=1) for i,r in enumerate([3.,-2.])]
        rows = [{'episode_key':(1,0,i),'return':g['reward']} for i,g in enumerate(games)]
        return controls,games,rows

    def test_baseline_is_frozen_control_not_other_candidate(self):
        controls,games,rows = self.fixtures()
        result=reference_advantages(controls,games,rows)
        self.assertEqual([r['baseline_return'] for r in rows],[1,1])
        self.assertEqual([r['advantage'] for r in rows],[.2,-.3])
        self.assertEqual(result['control_episodes'],1)
        games[1]['reward']=100.; rows[1]['return']=100.
        reference_advantages(controls,games,rows)
        self.assertEqual(rows[0]['advantage'],.2)

    def test_wrong_or_trainable_controls_are_rejected(self):
        controls,games,rows=self.fixtures()
        for bad in [[],controls*2,[dict(controls[0],frozen_control=False)],
                    [dict(controls[0],trainable_decisions=1)],[dict(controls[0],engine_seed=9)]]:
            with self.assertRaises(ValueError): reference_advantages(bad,games,rows)
        with self.assertRaises(ValueError): reference_advantages(controls,games,rows[:-1])

    def test_cpu_ppo_updates_actor_only_on_signed_advantages(self):
        torch.manual_seed(7)
        head=PointerHead(4,2); old=copy.deepcopy(head.state_dict())
        policy=SimpleNamespace(head=head,critic=torch.nn.Linear(4,1).requires_grad_(False),
                               config=self.config,model=SimpleNamespace(lm=torch.nn.Linear(1,1).requires_grad_(False)))
        rows=[]
        for i in range(4):
            decide,options=torch.randn(4),torch.randn(3,4)
            p=mixed_probabilities(head(decide,options),[0,2],.15).detach(); action=i%2
            rows.append({'decide':decide,'options':options,'eligible':[0,2],'reference':p,
                         'old_logp':float(p[action].log()),'action':action,'advantage':.2 if i%2 else -.3,'return':-10.})
        trainer=PPO(policy); loss=trainer.update(rows)
        self.assertEqual(loss['value_loss'],0)
        self.assertEqual(loss['critic_grad_norm'],0)
        self.assertEqual(len(trainer.optimizer.param_groups),1)
        self.assertTrue(any(not torch.equal(head.state_dict()[k],old[k]) for k in old))
        self.assertTrue(all(p.grad is None for p in policy.critic.parameters()))
        self.assertTrue(all(p.grad is None for p in policy.model.lm.parameters()))
        self.assertTrue(torch.isfinite(torch.tensor(trainer.reference_kl(rows))))

    def test_new_goal_and_early_stop(self):
        base={'mean_net_points':1000,'top2_rate':.5,'first_rate':.25,'fourth_rate':.25}
        self.assertEqual(development_gate(1,dict(base,mean_net_points=1500),base,self.config)[0],'target_reached')
        self.assertEqual(development_gate(2,dict(base,mean_net_points=1099),base,self.config)[0],'stop_no_early_progress')
        self.assertEqual(development_gate(2,dict(base,mean_net_points=1100),base,self.config)[0],'continue')
        self.assertEqual(development_gate(3,dict(base,mean_net_points=1100),base,self.config)[0],'stop_target_not_reached')

    def test_runner_collects_separate_controls_and_stops_at16(self):
        from scripts import train_head_rl as runner
        checkpoints,candidates,controls=[],[],[]
        class Policy:
            def __init__(self,config,resume):
                self.head=torch.nn.Linear(1,1); self.critic=torch.nn.Linear(1,1).requires_grad_(False)
                self.model=SimpleNamespace(lm=torch.nn.Linear(1,1).requires_grad_(False)); self.checks=32
            def game(self,seed,seat,training,old_opponent,replica,frozen=False):
                row={'seed':seed,'seat':seat,'replica':replica,'engine_seed':seed<<32,'game_mode':'4p-red-east',
                     'old_opponent':old_opponent,'rank':2,'reward':float(replica*2 if training else 1),
                     'score':25000+1000*(replica*2 if training else 1),'trainable_decisions':int(training)}
                if not training:
                    assert frozen and replica==-1; controls.append(row); return row,[]
                candidates.append(row)
                return row,[{'episode_key':(seed,seat,replica),'return':row['reward']}]
        class Trainer:
            def __init__(self,p): self.optimizer=torch.optim.AdamW(p.head.parameters())
            def update(self,rows): return {'loss':0.}
            def reference_kl(self,rows): return 0.
        def save(policy,trainer,out,update,*rest):
            checkpoints.append(update); p=out/'checkpoints'/f'update-{update:06d}'; p.mkdir(parents=True,exist_ok=True); return p
        def evaluate(policy,directory,c,deadline,frozen=False):
            rows=[{'seed':s,'seat':t,'rank':2,'score':26000} for s in range(c['development_seed_start'],c['development_seed_start']+c['development_seeds']) for t in range(4)]
            return summarize(rows),rows
        diagnostics={'sampled_states':64,'greedy_flips_vs_reference':0,'greedy_flip_fraction':0,'mean_entropy':1}
        with tempfile.TemporaryDirectory(dir=runner.ROOT/'runs',prefix='reference-test-') as temporary:
            parent=Path(temporary); ref=parent/'reference'; ref.mkdir()
            for name in ['head.pt','adapter_model.safetensors']: (ref/name).write_bytes(b'frozen')
            config=dict(self.config,out=str(parent/'trial'),reference=str(ref))
            p=parent/'config.json'; p.write_text(json.dumps(config))
            with patch('sys.argv',['train_head_rl.py','--config',str(p)]),patch.object(runner,'HeadPolicy',Policy), \
                 patch.object(runner,'PPO',Trainer),patch.object(runner,'save_checkpoint',side_effect=save), \
                 patch.object(runner,'evaluate',side_effect=evaluate) as ev,patch.object(runner,'rollout_diagnostics',return_value=diagnostics),patch('builtins.print'):
                runner.main()
            result=json.loads((Path(config['out'])/'pilot-analysis.json').read_text())
            self.assertEqual(result['updates'],16); self.assertEqual(result['stop_reason'],'stop_no_early_progress')
            self.assertEqual(len(candidates),64); self.assertEqual(len(controls),32)
            self.assertEqual(ev.call_count,3); self.assertEqual(checkpoints,[0,4,8,12,16])
            self.assertEqual(result['last_rollout']['reference_control_games'],32)
            self.assertEqual(result['critic_trainable_parameters'],0)


if __name__=='__main__': unittest.main()
