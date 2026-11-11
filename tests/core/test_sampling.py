import unittest
import copy
import json
import torch
from pathlib import Path
import tempfile
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
