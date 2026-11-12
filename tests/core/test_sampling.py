import unittest
import copy
import json
import torch
from pathlib import Path
import tempfile
from unittest.mock import patch
from tools.run_experiment import freeze_source
from jevmahjong.training.sampling import stratum, weights
from jevmahjong.data.native_pairs import ENCODER_FILES, load_sources, select_decisions
from jevmahjong.core.io import digest, request_hash
from jevmahjong.training.fit import initialize_heads, save_head_checkpoint, selection_metric, validation_guard
from jevmahjong.training.provenance import verify


def row(kind, delta):
    return {'category': kind, 'action': 1, 'teacher_action': 0, 'fast_advantage': delta,
        'public_request': {'questions': {'action': {'criteria': {'a0': {'type': 'none'}, 'a1': {'type': kind}}}}}}


def native_collection(root, start, gamma=.995):
    root.mkdir(); (root / 'walls-private').mkdir()
    config = {'game_mode': '4p-red-east', 'seed_scheme': 'spaced-v1', 'fast_gamma': gamma,
        'training_seed_start': start, 'training_seeds_per_stage': 4, 'training_replicas': 1,
        'intervention_categories': ['calling']}
    files = {role: {name: 'fixed-' + name for name in ENCODER_FILES} for role in ('discard', 'call')}
    (root / 'protocol.json').write_text(json.dumps({'config': config, 'model_files': files}))
    for seed in range(start, start + 4):
        events = [{'type': 'start_kyoku', 'scores': [25000] * 4, 'kyotaku': 0}, {'type': 'tsumo'},
            {'type': 'ryukyoku', 'deltas': [0] * 4}, {'type': 'end_kyoku'}, {'type': 'end_game'}]
        controls = [dict(seed=seed, seat=seat, replica=-1, score=25000, mjai_log=events,
            point_components={'draw_settlement': 0}, all_point_components=[{'draw_settlement': 0}] * 4,
            settlement_ledger_passed=True) for seat in range(4)]
        games = [dict(g, replica=0, trainable_decisions=int(g['seat'] == 0),
            reused_control_outcome=g['seat'] != 0) for g in controls]
        trace = dict(row('chi', 0.), category='calling', scope='calling', eligible=[0, 1],
            alternative_probability=1., episode_key=(seed, 0, 0), hand_index=0, draws_before=0,
            control_fast_reward=0., branch_fast_reward=0.)
        trace['request_sha256'] = request_hash(trace['public_request'])
        p = root / 'walls-private' / f'{seed}.pt'
        torch.save({'seed': seed, 'controls': controls, 'games': games, 'rows': [trace]}, p)
        p.with_suffix('.audit.json').write_text(json.dumps({'sha256': digest(p)}))
    (root / 'collection-result.json').write_text(json.dumps({'status': 'collection_complete', 'walls': 4, 'rows': 4}))
    return files


class SamplingTests(unittest.TestCase):
    def test_bounded_future_sampling_covers_rare_open_calls_without_using_outcomes(self):
        from jevmahjong.training.ev_collect import choose_cases,select_case_plan
        class Head:
            def __call__(self,decide,options):return torch.tensor([1.,0.])
        def case(index,opened):
            return dict(scope='calling',episode_key=(100,0,index),request_sha256=str(index),
                teacher_action=0,action=1,eligible=[0,1],decide=torch.zeros(1),options=torch.zeros(2,1),
                training_advantage=999. if index%2 else -999.,public_request={'state':{
                    'seat':0,'riichi':[],'melds':[[{'type':1}] if opened else [],[],[],[]]},
                    'questions':{'action':{'criteria':{'a0':{'type':'none'},'a1':{'type':'chi'}}}}})
        rows=[case(i,False) for i in range(10)]+[case(10,True)]
        parts={'train':rows,'held':[]}
        selected=choose_cases(parts,Head(),2,True,True)
        self.assertEqual([r['episode_key'][2] for r in selected],[0,10])
        for r in rows:r['training_advantage']*=-1
        self.assertEqual([r['episode_key'] for r in choose_cases(parts,Head(),2,True,True)],
            [r['episode_key'] for r in selected])
        self.assertEqual([r['episode_key'][2] for r in choose_cases(parts,Head(),2,True)],[0,1])
        with self.assertRaises(ValueError):choose_cases(parts,Head(),2,True,'yes')
        held=copy.deepcopy(rows)
        for r in held:r['episode_key']=(101,0,r['episode_key'][2])
        cfg=dict(max_states=5,max_states_by_split={'train':3,'held':2},canonical_state_only=True,
            stratify_call_openness=True)
        selected=select_case_plan({'train':rows,'held':held},Head(),cfg)
        self.assertEqual([r['source_split'] for r in selected],['train']*3+['held']*2)
        self.assertTrue(any(r['episode_key'][2]==10 for r in selected if r['source_split']=='held'))
        cfg['max_states']=6
        with self.assertRaises(ValueError):select_case_plan({'train':rows,'held':held},Head(),cfg)

    def test_queued_launch_uses_registered_source_without_resuming_model_state(self):
        from tools import run_experiment
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);(root/'jevmahjong').mkdir()
            source=root/'jevmahjong/__init__.py';source.write_text('version = 1\n')
            cfg=root/'config.json';cfg.write_text(json.dumps({'out':'runs/fit'}))
            with patch.object(run_experiment,'ROOT',root),patch.object(run_experiment.subprocess,'run') as invoke:
                with patch('sys.argv',['run_experiment','train-call-head','--config',str(cfg),'--registered']):
                    with self.assertRaises(ValueError):run_experiment.main()
                with patch('sys.argv',['run_experiment','train-call-head','--config',str(cfg),'--prepare-only']):
                    run_experiment.main()
                invoke.assert_not_called()
                source.write_text('version = 2\n')
                with patch('sys.argv',['run_experiment','train-call-head','--config',str(cfg),'--registered']):
                    run_experiment.main()
                args=invoke.call_args.args[0]
                self.assertNotIn('--resume',args)
                self.assertEqual((Path(args[3])/'jevmahjong/__init__.py').read_text(),'version = 1\n')

    def test_future_scope_is_explicit_and_preserves_legacy_kan_default(self):
        from jevmahjong.data.future_pairs import future_scope
        self.assertEqual(future_scope({}),('ordinary','kan_windows'))
        self.assertEqual(future_scope({'train_scopes':['calling'],'decision_filter':'all'}),('calling','all'))
        for invalid in [{'train_scopes':['calling']},
                {'train_scopes':['ordinary'],'decision_filter':'all'},
                {'train_scopes':['ordinary','calling'],'decision_filter':'all'}]:
            with self.assertRaises(ValueError):future_scope(invalid)

    def test_future_call_cases_keep_one_metadata_selected_origin_per_public_state(self):
        from jevmahjong.training.ev_collect import choose_cases
        req={'state':{'seat':0,'riichi':[1]},'questions':{'action':{'criteria':{
            'a0':{'type':'chi'},'a1':{'type':'pon'},'a2':{'type':'none'}}}}}
        base={'scope':'calling','public_request':req,'request_sha256':'same',
            'teacher_action':2,'action':0,'eligible':[0,1,2],
            'decide':torch.zeros(1),'options':torch.zeros(3,1),'episode_key':(100,0,1)}
        other=dict(base,action=1,episode_key=(100,0,2))
        class Head:
            def __call__(self,decide,options):return torch.tensor([-1.,-1.,1.])
        parts={'train':[other,base],'held':[]}
        self.assertEqual(len(choose_cases(parts,{'calling':Head()},None)),2)
        selected=choose_cases(parts,{'calling':Head()},None,True)
        self.assertEqual(len(selected),1);self.assertEqual(selected[0]['episode_key'],(100,0,1))
        self.assertEqual(selected[0]['teacher_action'],2)
        self.assertEqual(parts['train'][0]['action'],1)

    def test_same_state_alternatives_share_training_mass(self):
        from jevmahjong.training.sampling import weights
        first=dict(row('ankan',.1),episode_key=(100,0,0,1),request_sha256='same')
        first['public_request']['questions']['action']['criteria']['a2']={'type':'ankan'}
        second=copy.deepcopy(first);second.update(episode_key=(100,0,0,2),action=2)
        other=copy.deepcopy(first);other.update(episode_key=(101,0,0,1),request_sha256='other')
        w,report=weights([first,second,other],{'ankan':1.},4.,state_balanced=True)
        self.assertAlmostEqual(w[0]+w[1],w[2]);self.assertEqual(report['unique_states'],2)

    def test_option_expansion_keeps_all_eligible_actions_without_future_reward_selection(self):
        from jevmahjong.training.option_expand import representatives
        first={'episode_key':(100,0,1),'request_sha256':'same','teacher_action':0,'action':1,
            'eligible':[0,1,2],'training_advantage':-999.}
        second=dict(first,episode_key=(100,0,5),action=2,training_advantage=999.)
        selected=representatives([second,first])
        self.assertEqual([tuple(r['episode_key']) for r in selected],[(100,0,1)])
        self.assertEqual([(*selected[0]['episode_key'],a) for a in selected[0]['eligible'] if a!=0],
            [(100,0,1,1),(100,0,1,2)])

    def test_future_averages_remain_one_state_and_reject_changed_labels_seeds_and_points(self):
        from jevmahjong.data.future_pairs import mean_credit,validate_rows,METHOD
        from jevmahjong.training.collector import derived_seed
        from jevmahjong.core.scoring import settlement_ledger
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'walls-private').mkdir();source=root/'walls-private/100.pt'
            config={'source_root':str(root),'training_seed_start':100,'fast_gamma':.995,'seed':9,'worlds':2,
                'reward':{'mode':'net_points_with_fast_support_v1','net_weight':1.,'fast_weight':.1,
                    'dealin_penalty':.1,'point_scale':8000.}}
            start={'type':'start_kyoku','scores':[25000]*4,'kyotaku':0}
            old=dict(row('daiminkan',0.),scope='ordinary',episode_key=(100,0,0),eligible=[0,1],
                decide=torch.zeros(2),options=torch.zeros(2,2),action_keys=['none','daiminkan'],hand_index=0,draws_before=0)
            old['request_sha256']=request_hash(old['public_request'])
            control_log=[start,{'type':'ryukyoku','deltas':[0]*4},{'type':'end_kyoku'}]
            torch.save({'rows':[old],'controls':[{'seat':0,'mjai_log':control_log}],
                'games':[{'seed':100,'seat':0,'replica':0,'intervention_state_digest':'fixed'}]},source)
            def arm(log,scores):return {'mjai_log':log,'scores_at_stop':scores,
                'point_ledger':settlement_ledger(log,scores),'stopped_after_intervention_hand':True}
            branch_log=[start,{'type':'tsumo'},{'type':'hora','actor':0,'target':1,
                'deltas':[1000,-1000,0,0]},{'type':'end_kyoku'}]
            worlds=[{'future_seed':derived_seed(9,100,0,0,f'future-{i}'),'future_wall_sha256':'a'*64,
                'control':arm(control_log,[25000]*4),'branch':arm(branch_log,[26000,24000,25000,25000])} for i in range(2)]
            new=dict(old,reference=torch.tensor([.9,.1]),source_split='train',source_state_digest='fixed',
                prefix_events=1,counterfactual_worlds=worlds)
            means=mean_credit(worlds,new,config,config['reward']);new.update(training_credit=means,
                training_advantage=means['branch']['utility']-means['control']['utility'],
                fast_advantage=means['branch']['fast_reward']-means['control']['fast_reward'])
            checksum=digest(source);payload={'seed':100,'collection_method':METHOD,'source_path':str(source),
                'source_sha256':checksum,'rows':[new]};protocol={'source_files':{str(source):checksum},'selected_case_keys':[[100,0,0]]}
            validated=validate_rows(payload,100,config,protocol)
            self.assertEqual(len(validated),1);self.assertNotIn('counterfactual_worlds',validated[0])
            self.assertAlmostEqual(validated[0]['training_advantage'],.125+.1*.995)
            parent_root=root/'parent';(parent_root/'walls-private').mkdir(parents=True)
            parent=parent_root/'walls-private/100.pt';torch.save({'rows':[new]},parent)
            expanded=copy.deepcopy(new);expanded.update(episode_key=(100,0,0,1),origin_episode_key=(100,0,0),
                pair_selection='all_eligible_alternatives_against_play_v1')
            expanded_cfg=dict(config,expand_from=str(parent_root),parent_source_root=str(parent_root))
            expanded_protocol={'source_files':{str(source):checksum,str(parent):digest(parent)},'selected_case_keys':[[100,0,0,1]]}
            expanded_payload=dict(payload,rows=[expanded],parent_source_path=str(parent),parent_source_sha256=digest(parent))
            self.assertEqual(len(validate_rows(expanded_payload,100,expanded_cfg,expanded_protocol)),1)
            wrong=copy.deepcopy(expanded_payload);wrong['rows'][0]['counterfactual_worlds'][0]['future_wall_sha256']='b'*64
            with self.assertRaisesRegex(ValueError,'parent future'):validate_rows(wrong,100,expanded_cfg,expanded_protocol)
            changed=copy.deepcopy(payload);changed['rows'][0]['training_advantage']+=1
            with self.assertRaisesRegex(ValueError,'advantage changed'):validate_rows(changed,100,config,protocol)
            changed=copy.deepcopy(payload);changed['rows'][0]['counterfactual_worlds'][1]['future_seed']=0
            with self.assertRaisesRegex(ValueError,'Future seed'):validate_rows(changed,100,config,protocol)
            changed=copy.deepcopy(payload);changed['rows'][0]['counterfactual_worlds'][0]['branch']['scores_at_stop'][0]+=1
            with self.assertRaises(ValueError):validate_rows(changed,100,config,protocol)

    def test_kan_training_filter_keeps_pon_alternatives_and_whole_wall_assignment(self):
        regular=dict(row('dahai',1.),scope='ordinary',episode_key=(100,0,0))
        call=dict(row('pon',1.),scope='calling',episode_key=(101,0,0))
        kan=copy.deepcopy(call);kan['scope']='ordinary';kan['episode_key']=(103,0,0)
        kan['public_request']['questions']['action']['criteria']['a2']={'type':'daiminkan'}
        parts={'train':[regular,call],'held':[kan]}
        self.assertIs(select_decisions(parts),parts)
        selected=select_decisions(parts,'kan_windows',['ordinary'])
        self.assertEqual(selected['train'],[]);self.assertIs(selected['held'][0],kan)
        kan['fast_advantage']=-100.
        self.assertIs(select_decisions(parts,'kan_windows',['ordinary'])['held'][0],kan)
        self.assertEqual(select_decisions(parts,'all',['ordinary'])['train'],[regular])
        with self.assertRaises(ValueError):select_decisions(parts,'unknown')

    def test_risk_replay_increases_rare_context_without_copies_and_does_not_hide_regression(self):
        rows = [row('chi', delta) for delta in (1., -1.) for _ in range(5)]
        for r in rows: r['public_request']['state'] = {'seat': 0, 'riichi': [], 'melds': [[], [], [], []]}
        from riichienv import MeldType
        for i in (0, 5): rows[i]['public_request']['state'].update(riichi=[1], melds=[[{'type':int(MeldType.Pon)}],[],[],[]])
        values, report = weights(rows, {'chi': 1.}, 4., {'open_threat': .5, 'other_threat': .1, 'no_threat': .4})
        self.assertGreater(values[0], values[1]); self.assertLessEqual(max(values), 4.)
        self.assertGreater(report['effective_context_mass']['open_threat'], .2)
        parent = [{'risk_contexts': {'open_threat': {'loss': .2}}}]
        child = [{'risk_contexts': {'open_threat': {'loss': .3}}}]
        self.assertFalse(validation_guard(child, parent, ['open_threat']))
        child[0]['risk_contexts']['open_threat']['loss'] = .19
        self.assertTrue(validation_guard(child, parent, ['open_threat']))

    def test_changed_collection_behavior_checkpoint_is_rejected_before_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'native'; native_collection(source, 100)
            checkpoint = root / 'behavior.pt'; checkpoint.write_bytes(b'original')
            path = source / 'protocol.json'; protocol = json.loads(path.read_text())
            protocol['behavior_heads'] = {'checkpoint': str(checkpoint), 'sha256': digest(checkpoint)}
            path.write_text(json.dumps(protocol))
            _, _, sources = load_sources(source)
            self.assertEqual(sources[0]['behavior_heads'], protocol['behavior_heads'])
            checkpoint.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'behavior head checkpoint changed'):
                load_sources(source)

    def test_new_source_regression_is_not_hidden_by_old_validation_improvement(self):
        parent = [{'source': 'old', 'rows': 259, 'loss': .1731035024},
            {'source': 'new', 'rows': 328, 'loss': .1653106644}]
        child = [{'source': 'old', 'rows': 259, 'loss': .1681632703},
            {'source': 'new', 'rows': 328, 'loss': .1670793201}]
        self.assertLess(selection_metric(child, 'all_sources'), selection_metric(parent, 'all_sources'))
        self.assertGreater(selection_metric(child, 'latest_source'), selection_metric(parent, 'latest_source'))
        self.assertEqual(selection_metric(parent[-1:], 'all_sources'), selection_metric(parent, 'latest_source'))
        with self.assertRaises(ValueError): selection_metric(parent, 'unknown')

    def test_continuation_restores_parent_heads_and_preserves_epoch_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); source = root / 'parent.pt'
            parent = {s: torch.nn.Linear(2, 2, bias=False) for s in ('ordinary', 'calling')}
            with torch.no_grad():
                for head in parent.values(): head.weight.fill_(.75)
            files = {'discard': {}, 'call': {}}
            torch.save({'model_id': 'jev-tempo-safe-v1', 'heads': {s: h.state_dict() for s,h in parent.items()},
                'model_files': files, 'base_and_LoRA_frozen': True}, source)
            protocol = {'model_files': files, 'code_sha256': {},
                'initial_heads': {'checkpoint': str(source), 'sha256': digest(source)}}
            heads = {s: torch.nn.Linear(2, 2, bias=False) for s in parent}
            info = initialize_heads(heads, protocol)
            self.assertTrue(info['optimizer_reinitialized'])
            item = save_head_checkpoint(root, 'child', heads, protocol, 0, .2)
            zero = torch.load(item['checkpoint'], map_location='cpu', weights_only=True)
            for scope in heads:
                self.assertTrue(torch.equal(zero['heads'][scope]['weight'], parent[scope].weight))
            verify(protocol)
            source.write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                initialize_heads(heads, protocol)
            with self.assertRaisesRegex(ValueError, 'checkpoint changed'):
                verify(protocol)

    def test_combined_collections_keep_all_seats_and_branches_in_whole_wall_split(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / 'a', Path(directory) / 'b'
            files = native_collection(a, 100); native_collection(b, 104)
            partitions, hashes, sources = load_sources([a, b], files)
            self.assertEqual([r['episode_key'][0] for r in partitions['held']], [103, 107])
            self.assertEqual(len(partitions['train']), 6)
            self.assertEqual(len(hashes), 8)
            self.assertEqual(sum(s['train_walls'] for s in sources), 6)
            utility = {'mode': 'risk_adjusted_fast_win_v1', 'dealin_penalty': .3, 'net_weight': .1, 'point_scale': 8000.}
            relabeled, _, _ = load_sources([a, b], files, utility)
            old, new = partitions['train'][0], relabeled['train'][0]
            self.assertEqual(old['public_request'], new['public_request'])
            self.assertEqual(old['request_sha256'], new['request_sha256'])
            self.assertNotIn('training_advantage', old)
            self.assertEqual(new['training_advantage'], 0.)
            with self.assertRaisesRegex(ValueError, 'Unique'):
                load_sources([a, a])
            different = copy.deepcopy(files); different['call']['adapter_config.json'] = 'wrong'
            with self.assertRaisesRegex(ValueError, 'encoders'):
                load_sources([a], different)

    def test_duplicate_walls_reward_mismatch_and_tampered_shards_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b, c = (Path(directory) / n for n in ('a', 'b', 'c'))
            native_collection(a, 100); native_collection(b, 100); native_collection(c, 104, .99)
            with self.assertRaisesRegex(ValueError, 'Whole wall repeated'):
                load_sources([a, b])
            with self.assertRaisesRegex(ValueError, 'compatible'):
                load_sources([a, c])
            (a / 'walls-private/100.pt').write_bytes(b'changed')
            with self.assertRaisesRegex(ValueError, 'shard changed'):
                load_sources(a)

    def test_complete_marker_cannot_hide_missing_registered_walls(self):
        with tempfile.TemporaryDirectory() as directory:
            a = Path(directory) / 'a'; native_collection(a, 100)
            (a / 'walls-private/100.pt').unlink()
            with self.assertRaisesRegex(ValueError, 'coverage differs'):
                load_sources(a)

    def test_editing_main_code_keeps_existing_run_snapshot_unchanged(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); package = root / 'jevmahjong'; package.mkdir()
            file = package / '__init__.py'; file.write_text('VERSION = 1\n')
            first = freeze_source(root)
            file.write_text('VERSION = 2\n')
            second = freeze_source(root)
            self.assertNotEqual(first, second)
            self.assertEqual((first / 'jevmahjong/__init__.py').read_text(), 'VERSION = 1\n')
            self.assertEqual((second / 'jevmahjong/__init__.py').read_text(), 'VERSION = 2\n')

    def test_known_mass_and_direction_are_balanced_without_copying_rows(self):
        rows = [row('chi', 1.) for _ in range(3)] + [row('chi', -1.)] + [row('pon', 1.), row('pon', -1.)]
        w, report = weights(rows, {'chi': .5, 'pon': .5}, 4.)
        self.assertAlmostEqual(sum(w[:4]), sum(w[4:]))
        self.assertAlmostEqual(sum(w[:3]), w[3])
        self.assertEqual(report['copies_added'], 0)
        self.assertIn('ankan/positive', report['missing_action_directions'])

    def test_skipping_beneficial_operation_reverses_label_and_rare_weight_is_capped(self):
        r = row('daiminkan', -1.); r['teacher_action'], r['action'] = 1, 0
        self.assertEqual(stratum(r), ('daiminkan', 'positive'))
        rows = [row('chi', 1.) for _ in range(30)] + [row('daiminkan', 1.)]
        w, report = weights(rows, {'chi': .5, 'daiminkan': .5}, 4.)
        self.assertEqual(max(w), 4.)
        self.assertLess(report['effective_action_mass']['daiminkan'], .5)
        r = row('pon', 1.); r['category'] = 'chi_threat'
        r['public_request']['questions']['action']['criteria']['a0']['type'] = 'chi'
        self.assertEqual(stratum(r), ('chi', 'negative'))


if __name__ == '__main__': unittest.main()
