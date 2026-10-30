import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from diagonal_replication import shard_configs
from head_rl import write
from scripts.audit_score_confirmation import compare,sha256
from scripts.finish_diagonal_replication import finish


class FullReplicationAssemblyTests(unittest.TestCase):
    def test_complete_fixed_sample_is_joined_audited_and_recoverable(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);out=root/'replication';out.mkdir()
            for name in ('reference','candidate'):
                checkpoint=root/name;checkpoint.mkdir()
                (checkpoint/'head.pt').write_bytes(name.encode())
                (checkpoint/'adapter_model.safetensors').write_bytes(b'same-frozen-adapter')
            config={'out':str(out),'reference':'reference','confirmation_seed_start':10,'confirmation_seeds':512,
                    'development_seed_start':5000,'development_seeds':2,'reward_scale':1000.,'reward_mode':'net_points',
                    'game_mode':'4p-red-east','seed_scheme':'spaced-v1','target_mean_net_points':0,'target_net_points_gain':0}
            expected=sha256(root/'candidate/head.pt')
            for cfg in [config,*shard_configs(config)]:
                directory=Path(cfg['out']);directory.mkdir(exist_ok=True)
                protocol={'config':cfg,'sha256':hashlib.sha256(json.dumps(cfg,sort_keys=True).encode()).hexdigest(),
                          'reference_head_sha256':sha256(root/'reference/head.pt'),
                          'reference_adapter_sha256':sha256(root/'reference/adapter_model.safetensors'),
                          'code_sha256':{},'shards_predeclared':2}
                write(directory/'protocol.json',protocol)
                write(directory/'confirmation-selection.json',{'config':cfg,'checkpoint':'candidate','candidate_head_sha256':expected})
                write(directory/'training-games-stage-01.json',[{'seed':1}])
                if directory==out:continue
                records={}
                for name in ('reference','candidate'):
                    scores=25000 if name=='reference' else 25100
                    rows=[{'seed':s,'seat':seat,'engine_seed':s*(1<<32),'seed_scheme':'spaced-v1',
                           'game_mode':'4p-red-east','reward_mode':'net_points','reward_scale':1000.,
                           'score':scores,'rank':2,'reward':(scores-25000)/1000.,'sampling_seed':None,
                           'trainable_decisions':0,'old_opponent':False}
                          for s in range(cfg['confirmation_seed_start'],cfg['confirmation_seed_start']+cfg['confirmation_seeds'])
                          for seat in range(4)]
                    arm=directory/f'confirmation-{name}';arm.mkdir()
                    (arm/'games.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows));records[name]=rows
                checked=compare(records['candidate'],records['reference'],cfg)
                write(directory/'confirmation-result.json',{'status':'complete','checkpoint':'candidate','default_policy_changed':False,
                     'candidate':checked['candidate'],'reference':checked['reference'],'primary_metric':'mean_net_points',
                     'qualified':checked['qualified'],'primary_delta_ci95':checked['primary_delta_ci95'],
                     'paired':{'primary_delta':checked['primary_delta']}})
            with patch('scripts.finish_diagonal_replication.ROOT',root),patch('scripts.audit_score_confirmation.ROOT',root):
                report=finish(out)
                self.assertEqual(report['audit_status'],'passed');self.assertEqual(report['primary_delta'],100.)
                self.assertEqual(report['candidate']['games'],2048);self.assertEqual(report['candidate']['independent_seeds'],512)
                self.assertEqual(len(report['shard_integrity']),2)
                digest=sha256(out/'confirmation-candidate/games.jsonl')
                (out/'confirmation-audit.json').unlink()  # interrupted after result, before audit persistence
                repeated=finish(out)
                self.assertEqual(repeated['primary_delta'],100.)
                self.assertTrue((out/'confirmation-audit.json').is_file())
                self.assertEqual(digest,sha256(out/'confirmation-candidate/games.jsonl'))


if __name__=='__main__':unittest.main()
