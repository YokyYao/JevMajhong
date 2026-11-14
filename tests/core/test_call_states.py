"""Every-call native inputs remain unlabeled and disjoint from kan routing."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from tests.core.test_defense_states import fixture
from jevmahjong.core.io import digest,request_hash
from jevmahjong.data.native_states import CALL_METHOD,decision_records,validate
from jevmahjong.training.call_states import CallStateCollector


def call_fixture():
    p=fixture();p['collection_method']=CALL_METHOD;r=p['rows'][0]
    r['scope']='calling';r['eligible']=[0,1,2]
    r['public_request']['questions']['action']['criteria']={
        'a0':{'type':'chi'},'a1':{'type':'pon'},'a2':{'type':'none'}}
    r['request_sha256']=request_hash(r['public_request']);return p


class CallStatesTests(unittest.TestCase):
    def test_open_and_closed_call_inputs_keep_full_menu_without_invented_rewards(self):
        for melds in [[],[{'type':1}]]:
            p=call_fixture();r=p['rows'][0];r['public_request']['state']['melds'][0]=melds
            r['request_sha256']=request_hash(r['public_request'])
            self.assertEqual(validate(p,10,{})[0]['eligible'],[0,1,2])
            self.assertEqual(decision_records(p),p['decisions'])
            self.assertNotIn('training_advantage',r);self.assertNotIn('games',p)
        for mutation in ['scope','kan','eligible']:
            p=call_fixture();r=p['rows'][0]
            if mutation=='scope':r['scope']='ordinary'
            if mutation=='eligible':r['eligible']=[0,1]
            if mutation=='kan':r['public_request']['questions']['action']['criteria']['a1']['type']='daiminkan'
            r['request_sha256']=request_hash(r['public_request'])
            with self.subTest(mutation=mutation),self.assertRaises(ValueError):validate(p,10,{})

    def test_collector_keeps_every_call_and_excludes_ordinary_without_changing_action(self):
        collector=CallStateCollector.__new__(CallStateCollector)
        def act(instance,obs,frozen,stochastic,trace):
            if trace is not None:trace.extend(obs)
            return 'same-native-action'
        with patch('jevmahjong.training.collector.NativeCollector.act',act):
            for scope in ['calling','ordinary']:
                trace=[];choice=collector.act([{'scope':scope}],frozen=True,trace=trace)
                self.assertEqual(choice,'same-native-action')
                self.assertEqual(len(trace),int(scope=='calling'))
            self.assertEqual(collector.act([],frozen=True),'same-native-action')

    def test_call_inputs_require_real_labels_before_fit_and_keep_whole_wall_split(self):
        from jevmahjong.data.native_pairs import load_sources,ENCODER_FILES
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'walls-private').mkdir()
            cfg=dict(game_mode='4p-red-east',seed_scheme='spaced-v1',fast_gamma=.995,
                training_seed_start=10,training_seeds_per_stage=4)
            files={role:{name:'fixed-'+name for name in ENCODER_FILES} for role in ['discard','call']}
            (root/'protocol.json').write_text(json.dumps(dict(config=cfg,model_files=files,collection_method=CALL_METHOD)))
            for seed in range(10,14):
                p=call_fixture();p['seed']=seed
                for c in p['controls']:c['seed']=seed
                p['rows'][0]['episode_key']=(seed,0,0);p['decisions'][0].update(seed=seed,episode_key=(seed,0,0))
                path=root/'walls-private'/f'{seed}.pt';torch.save(p,path)
                path.with_suffix('.audit.json').write_text(json.dumps({'sha256':digest(path)}))
            (root/'collection-result.json').write_text(json.dumps(dict(status='collection_complete',walls=4,rows=4)))
            with self.assertRaisesRegex(ValueError,'no reward labels'):load_sources(root)
            parts,_,_=load_sources(root,allow_unlabeled=True)
            self.assertEqual([len(parts[k]) for k in ['train','held']],[3,1])
            self.assertTrue(all(r['scope']=='calling' for rs in parts.values() for r in rs))

    def test_future_reader_accepts_call_metadata_and_rejects_defense_scope(self):
        from jevmahjong.data.future_pairs import validate_rows,METHOD
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);(root/'walls-private').mkdir();path=root/'walls-private/10.pt'
            torch.save(call_fixture(),path);sha=digest(path)
            payload=dict(seed=10,collection_method=METHOD,source_path=str(path),source_sha256=sha,rows=[])
            protocol=dict(source_files={str(path):sha},selected_case_keys=[])
            cfg=dict(train_scopes=['calling'],decision_filter='all',source_root=str(root),training_seed_start=10)
            self.assertEqual(validate_rows(payload,10,cfg,protocol),[])
            cfg.update(train_scopes=['ordinary'],decision_filter='open_tenpai_riichi')
            with self.assertRaisesRegex(ValueError,'declared scope'):validate_rows(payload,10,cfg,protocol)


if __name__=='__main__':unittest.main()
