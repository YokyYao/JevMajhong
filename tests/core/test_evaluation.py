"""Known paired outcomes must drive score/speed decisions on the complete grid."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
import torch
from jevmahjong.evaluation.action_suite import evaluate as action_panel
from jevmahjong.evaluation.metrics import summarize
from jevmahjong.evaluation.reference_cache import GAMEPLAY_FILES, load_reference
from jevmahjong.evaluation.run import candidate_scopes, select_heads, candidate_activation, parity_sources, parity_anchor_paths
from jevmahjong.policy.kan import KanPolicyClient
from jevmahjong.core.io import digest


def example():
    cfg = {'seed_start': 100, 'seeds': 4, 'stage': 'development', 'fast_gamma': .995}
    rows = []
    for seed in range(100, 104):
        for suite in ('closed', 'mixed', 'strong'):
            for seat in range(4):
                for policy in ('reference', 'candidate'):
                    changed = policy == 'candidate'
                    scores = [25000] * 4
                    delta = [0] * 4
                    if changed:
                        delta[seat] = 100; delta[(seat + 1) % 4] = -100
                        scores = [a + b for a, b in zip(scores, delta)]
                    events = [{'type': 'start_kyoku', 'scores': [25000] * 4, 'kyotaku': 0}]
                    events += [{'type': 'tsumo'} for _ in range(30)]
                    events += [{'type': 'hora', 'actor': seat, 'target': (seat + 1) % 4, 'deltas': delta}] if changed else [
                        {'type': 'ryukyoku', 'deltas': delta}]
                    events += [{'type': 'end_kyoku'}, {'type': 'end_game'}]
                    rows.append(dict(seed=seed, suite=suite, seat=seat, policy=policy,
                        score=25100 if changed else 25000, rank=1, hands=1,
                        hand_wins=int(changed), dealins=0, calls=int(changed),
                        point_components={'win_income': 100} if changed else {'draw_settlement': 0},
                        settlement_ledger_passed=True, fast_hand_count=1, fast_hand_wins=int(changed),
                        fast_reward_sum=.995 ** 30 if changed else 0., winning_table_draws=30 if changed else 0,
                        mjai_log=events, final_scores=scores,
                        kan_opportunities={'ankan': 1}, kan_choices={'ankan': int(changed)}))
    return rows, cfg


class OpenTenpaiAdmissionTest(unittest.TestCase):
    @staticmethod
    def case(threats=(1,), own_riichi=False, opened=True):
        from types import SimpleNamespace
        class Observation:
            player_id=0
            def to_dict(self):
                return {'riichi_declared':[own_riichi,1 in threats,2 in threats,3 in threats],
                    'melds':[[SimpleNamespace(opened=opened)],[],[],[]]}
        actions=[SimpleNamespace(action_type=None),SimpleNamespace(action_type=None)]
        features=[{'sh':0,'safe':[],'ukeire':8,'red':False},
            {'sh':1,'safe':[1],'ukeire':2,'red':False}]
        return Observation(),actions,features

    def test_open_tenpai_can_consider_safe_discard_that_breaks_tenpai(self):
        from jevmahjong.evaluation.agents import guarded_indices
        from tools.open_tenpai_admission import eligible_indices
        obs,actions,features=self.case()
        self.assertEqual(guarded_indices(obs,actions,features),[0])
        self.assertEqual(eligible_indices(obs,actions,features),[0,1])
        # A tile safe against one riichi must not be treated as safe against two.
        obs,actions,features=self.case(threats=(1,2))
        self.assertEqual(eligible_indices(obs,actions,features),[0])
        features[1]['safe']=[1,2]
        self.assertEqual(eligible_indices(obs,actions,features),[0,1])

    def test_other_hand_states_keep_the_existing_guard(self):
        from jevmahjong.evaluation.agents import guarded_indices
        from tools.open_tenpai_admission import eligible_indices
        for kwargs in [{'threats':()},{'own_riichi':True},{'opened':False}]:
            with self.subTest(**kwargs):
                obs,actions,features=self.case(**kwargs)
                self.assertEqual(eligible_indices(obs,actions,features),guarded_indices(obs,actions,features))
        obs,actions,features=self.case()
        features[0]['sh']=1;features[1]['sh']=2
        self.assertEqual(eligible_indices(obs,actions,features),guarded_indices(obs,actions,features))

    def test_adding_safe_discard_keeps_existing_non_discard_options(self):
        from types import SimpleNamespace
        from tools.open_tenpai_admission import eligible_indices
        obs,actions,features=self.case()
        actions.append(SimpleNamespace(action_type=None));features.append(None)
        self.assertEqual(eligible_indices(obs,actions,features),[0,1,2])


class FastEvaluationTest(unittest.TestCase):
    def test_native_parity_collection_is_accepted_without_a_fit_result(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);shards=root/'walls-private';shards.mkdir()
            shard=shards/'100.pt';torch.save({'rows':[{}]},shard)
            (root/'protocol.json').write_text(json.dumps({'config':{'training_seed_start':100,'training_seeds_per_stage':1},
                'model_files':{},'code_sha256':{}}))
            (root/'collection-result.json').write_text(json.dumps({'status':'collection_complete','walls':1,'rows':1}))
            receipt=shard.with_suffix('.audit.json');receipt.write_text(json.dumps({'sha256':digest(shard),'rows':1}))
            files=parity_anchor_paths(root,{})
            self.assertIn(str(shard),files);self.assertFalse((root/'result.json').exists())
            shard.write_bytes(b'changed')
            with self.assertRaises(ValueError):parity_anchor_paths(root,{})

    def test_hand_win_objective_reports_point_loss_without_calling_it_a_win_rate_loss(self):
        rows,cfg=example();cfg.update(confirmation_objective='hand_win',max_dealin_rate_increase=.01)
        for r in rows:
            seat=r['seat'];scores=list(r['final_scores']);other=(seat+1)%4
            delta=[0]*4
            if r['policy']=='candidate':
                delta=[-600]*4;delta[other]=1800
                ending={'type':'hora','actor':other,'target':other,'deltas':delta}
                r['point_components']['tsumo_payments']=-600
            else:ending={'type':'ryukyoku','deltas':delta}
            r['mjai_log']=r['mjai_log'][:-1]+[{'type':'start_kyoku','scores':scores,'kyotaku':0}]+[
                {'type':'tsumo'}]*4+[ending,{'type':'end_kyoku'},{'type':'end_game'}]
            r['final_scores']=[a+b for a,b in zip(scores,delta)];r['score']=r['final_scores'][seat]
            r['rank']=sorted(range(4),key=lambda p:(-r['final_scores'][p],p)).index(seat)+1
            r['hands']=r['fast_hand_count']=2
        result=summarize(rows,cfg)
        self.assertEqual(result['hand_win_paired']['gain'],.5)
        self.assertEqual(result['overall']['candidate_minus_reference']['gain'],-500.)
        self.assertTrue(result['positive_signal']);self.assertTrue(result['eligible_for_confirmation'])
        cfg['confirmation_objective']='net_points'
        self.assertFalse(summarize(rows,cfg)['eligible_for_confirmation'])

    def test_kan_only_ev_data_keeps_pinned_inputs_for_unchanged_route_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);protocol=root/'protocol.json'
            protocol.write_text(json.dumps({'collection_method':'paired_future_ev_v1','source_files':{'original.pt':'original-hash'}}))
            fit={'data_sources':[{'path':str(root),'protocol_sha256':digest(protocol)}],
                'data_sha256':{'revalued.pt':'new-hash'}}
            self.assertEqual(list(parity_sources(fit)),['original.pt','revalued.pt'])
            protocol.write_text('{}')
            # A changed method marker cannot bypass the stored provenance hash.
            with self.assertRaises(ValueError):parity_sources(fit)

    def test_kan_activation_preserves_discard_and_pure_call_routes(self):
        class Client:
            def __init__(self, name): self.name=name
            def system_one(self, **request): return self.name
        router=object.__new__(KanPolicyClient);router.frozen=Client('play');router.updated=Client('kan')
        for kinds, expected in [(['dahai','reach'],'play'),(['chi','pon','none'],'play'),
                (['dahai','ankan'],'kan'),(['pon','daiminkan','none'],'kan'),(['dahai','kakan'],'kan')]:
            req={'questions':{'action':{'criteria':{str(i):{'type':kind} for i,kind in enumerate(kinds)}}}}
            self.assertEqual(router.system_one(**req),expected)
        self.assertEqual(candidate_activation({'updated_scopes':['ordinary'],'candidate_activation':'kan_windows'}),'kan_windows')
        with self.assertRaises(ValueError):candidate_activation({'candidate_activation':'kan_windows'})
        with self.assertRaises(ValueError):candidate_activation({'candidate_activation':'unknown'})
        with self.assertRaises(ValueError):candidate_activation({'updated_scopes':['ordinary']},'kan_windows')

    def test_head_ablation_keeps_unselected_reference_parameters_and_contract_explicit(self):
        config={'updated_scopes':['calling'],'candidate_activation':'call_windows'}
        self.assertEqual(candidate_activation(config,'call_windows'),'call_windows')
        with self.assertRaises(ValueError):candidate_activation(dict(config,updated_scopes=['ordinary']))
        with self.assertRaises(ValueError):candidate_activation(dict(config,candidate_activation='all'),'call_windows')
        with self.assertRaises(ValueError):candidate_activation(config,'kan_windows')
        for scope in ('ordinary', 'calling'):
            heads = {s:torch.nn.Linear(1, 1, bias=False) for s in ('ordinary', 'calling')}
            with torch.no_grad():
                for h in heads.values():h.weight.fill_(1.)
            saved = {s:{'weight':torch.tensor([[3.]])} for s in heads}
            select_heads(heads, saved, {'updated_scopes':[scope]})
            self.assertEqual(float(heads[scope].weight.detach()), 3.)
            other = 'calling' if scope == 'ordinary' else 'ordinary'
            self.assertEqual(float(heads[other].weight.detach()), 1.)
        self.assertEqual(candidate_scopes({}), ('ordinary','calling'))
        self.assertNotEqual(candidate_scopes({'updated_scopes':['ordinary']}), candidate_scopes({}))
        for invalid in ([], ['ordinary','ordinary'], ['unknown'], 'ordinary'):
            with self.assertRaises(ValueError):candidate_scopes({'updated_scopes':invalid})

    def test_multiple_ron_is_one_dealin_hand_and_two_payment_events(self):
        rows, cfg = example(); r = rows[0]; seat = r['seat']
        first, second = [0] * 4, [0] * 4
        first[seat] = second[seat] = -100; first[1] = second[2] = 100
        r.update(score=24800, rank=4, dealins=2, point_components={'ron_payments': -200},
            final_scores=[24800, 25100, 25100, 25000],
            mjai_log=[{'type': 'start_kyoku', 'scores': [25000] * 4, 'kyotaku': 0}, {'type': 'tsumo'},
                {'type': 'hora', 'actor': 1, 'target': seat, 'deltas': first},
                {'type': 'hora', 'actor': 2, 'target': seat, 'deltas': second}, {'type': 'end_kyoku'}, {'type': 'end_game'}])
        result = summarize(rows, cfg)['overall']['policies']['reference']
        self.assertAlmostEqual(result['dealin_rate'] * result['games'], 1)
        self.assertAlmostEqual(result['dealin_event_rate'] * result['games'], 2)

    def test_reference_cache_requires_complete_grid_matching_contract_and_unchanged_logs(self):
        rows, cfg = example()
        package = Path(__file__).resolve().parents[2] / 'jevmahjong'
        code = {str(package / p): digest(package / p) for p in GAMEPLAY_FILES}
        protocol = {'config': cfg, 'code_sha256': code, 'model_files': {'discard': {}, 'call': {}}}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory); raw = root / 'games-private.jsonl'
            raw.write_text(''.join(json.dumps(r) + '\n' for r in rows))
            (root / 'protocol.json').write_text(json.dumps(protocol))
            (root / 'result.json').write_text(json.dumps(summarize(rows, cfg)))
            (root / 'audit.json').write_text(json.dumps({'status': 'passed_complete',
                'games': len(rows), 'games_sha256': digest(raw)}))
            cached, receipt = load_reference(root, cfg, protocol)
            self.assertEqual(len(cached), len(rows) // 2)
            self.assertTrue(all(r['policy'] == 'reference' for r in cached))
            self.assertEqual(receipt['games_sha256'], digest(raw))
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                load_reference(root, dict(cfg, seed_start=200), protocol)
            changed = copy.deepcopy(protocol); changed['model_files']['call']['head.pt'] = 'changed'
            with self.assertRaisesRegex(ValueError, 'contract differs'):
                load_reference(root, cfg, changed)
            raw.write_text(raw.read_text() + '\n')
            with self.assertRaisesRegex(ValueError, 'unchanged'):
                load_reference(root, cfg, protocol)

    def test_declared_point_objective_does_not_reject_point_gain_for_slower_wins(self):
        rows, cfg = example()
        for r in rows:
            if r['policy'] == 'reference':
                r.update(hand_wins=1, fast_hand_wins=1, fast_reward_sum=.995, winning_table_draws=1,
                    point_components={'win_income': 0},
                    mjai_log=[{'type': 'start_kyoku', 'scores': [25000] * 4, 'kyotaku': 0},
                        {'type': 'tsumo'}, {'type': 'hora', 'actor': r['seat'], 'target': (r['seat'] + 1) % 4,
                        'deltas': [0] * 4}, {'type': 'end_kyoku'}, {'type': 'end_game'}])
        self.assertFalse(summarize(rows, cfg)['eligible_for_confirmation'])
        cfg['confirmation_objective'] = 'net_points'
        result = summarize(rows, cfg)
        self.assertLess(result['fast_reward_paired']['gain'], 0)
        self.assertGreater(result['overall']['candidate_minus_reference']['gain'], 0)
        self.assertTrue(result['eligible_for_confirmation'])

    def test_probability_improvement_does_not_count_as_action_success(self):
        class FixedHead:
            def __call__(self, decide, options):
                return torch.tensor([.8, .2]).log()
        row = {'category': 'chi', 'scope': 'calling', 'action': 1, 'teacher_action': 0,
            'fast_advantage': 1., 'episode_key': (1, 0, 0), 'eligible': [0, 1],
            'decide': None, 'options': None, 'reference': torch.tensor([.95, .05]),
            'public_request': {'questions': {'action': {'criteria': {
                'a0': {'type': 'none'}, 'a1': {'type': 'chi'}}}}}}
        panel = action_panel({'calling': FixedHead()}, [row], 1)['actions']['chi']
        positive = panel['positive']
        self.assertEqual(positive['pair_order_correct'], 0)
        self.assertEqual(positive['operation_selected_from_full_legal_set'], 0)
        self.assertAlmostEqual(positive['candidate_operation_probability_mean'], .2)
        self.assertGreater(positive['reward_signed_pair_margin_gain_mean'], 0)
        self.assertFalse(panel['negative']['coverage_complete'])
        self.assertIsNone(panel['negative']['reward_signed_pair_margin_gain_mean'])

    def test_known_paired_speed_and_points_use_complete_whole_walls(self):
        rows, cfg = example(); result = summarize(rows, cfg)
        self.assertAlmostEqual(result['fast_reward_paired']['gain'], .995 ** 30)
        self.assertEqual(result['overall']['candidate_minus_reference']['gain'], 100)
        self.assertTrue(result['eligible_for_confirmation'])
        self.assertEqual(result['overall']['policies']['candidate']['kan_choices']['ankan'], 48)

    def test_speed_alone_with_point_loss_does_not_trigger_confirmation(self):
        rows, cfg = example()
        for r in rows:
            if r['policy'] == 'candidate':
                seat=r['seat']; other=(seat+1)%4; delta=[0]*4; delta[seat]=-200;delta[other]=200
                scores=[a+b for a,b in zip(r['final_scores'],delta)]
                r['mjai_log'] = r['mjai_log'][:-1] + [
                    {'type':'start_kyoku','scores':r['final_scores'],'kyotaku':0},
                    {'type':'hora','actor':other,'target':seat,'deltas':delta},
                    {'type':'end_kyoku'},{'type':'end_game'}]
                r.update(score=24900, point_components={'win_income':100,'ron_payments':-200},
                    final_scores=scores,hands=2,fast_hand_count=2,dealins=1)
        self.assertFalse(summarize(rows, cfg)['eligible_for_confirmation'])

    def test_missing_duplicate_and_wrong_hand_credit_rejected(self):
        rows, cfg = example()
        with self.assertRaises(ValueError):
            summarize(rows[:-1], cfg)
        with self.assertRaises(ValueError):
            summarize(rows + [rows[0]], cfg)
        rows[0]['fast_hand_count'] = 2
        with self.assertRaises(ValueError):
            summarize(rows, cfg)


if __name__ == '__main__':
    unittest.main()
