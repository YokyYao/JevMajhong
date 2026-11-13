"""Own-win model promotion, credit accounting, and independent fit verification."""
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from jevmahjong.core.io import digest
from jevmahjong.evaluation.context_audit import (
    audit, check_profiles, manual_win_improves, WIN_METRIC,
)
from jevmahjong.policy.kan_context import feature_names
from jevmahjong.policy.kan_progress import VERSION
from jevmahjong.training.context_policy import (
    fit, pack_options, policy_profile, selection_profile, training_groups, win_policy_improved,
)
from tests.core.test_kan_context import request


def selection(win=0., fast=0., risk=0., utility=0., points=0.):
    return dict(mean_sampled_hand_win_gain_per_state=win,
                mean_sampled_fast_reward_change_per_state=fast,
                mean_sampled_dealin_change_per_state=risk,
                mean_label_utility_gain_vs_play=utility,
                mean_sampled_net_point_gain_per_state=points)


def pair(seed):
    req = request('daiminkan')
    req['state']['rivers'][1] = 'E'
    req['questions']['action']['criteria']['a1'] = dict(type='pon', pai='E', consumed=['E', 'E'])
    q = torch.tensor([.9, .1])
    control = dict(utility=.175, own_win=.25, fast_reward=.1, dealin=.05, net_points=1000.)
    branch = dict(utility=.3125, own_win=.375, fast_reward=.15, dealin=.05, net_points=500.)
    return dict(episode_key=(seed, 0, 0, 1), request_sha256='same-public-state', scope='calling',
                public_request=req, eligible=[0, 1], teacher_action=0, action=1,
                reference=q, decide=torch.zeros(1), options=q.log(),
                training_advantage=.1375, training_credit=dict(control=control, branch=branch))


class FrozenHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.zeros(2))

    def forward(self, decide, options):
        return options + self.offset


class WinContextTests(unittest.TestCase):
    def test_capacity_checkpoint_roundtrip_matches_actual_inference(self):
        from jevmahjong.training.context_network import make_network
        from jevmahjong.policy.kan_context import context_scores, probabilities
        req=pair(100)['public_request'];names=feature_names(VERSION)
        net=make_network(256,len(names))
        self.assertEqual(sum(p.numel() for p in net.parameters()),40960)
        context=dict(version=VERSION,names=list(names),model_kind='mlp',hidden=256,
                     state_dict=net.state_dict(),reference_prior_strength=.05)
        self.assertEqual(probabilities(req,[.9,.1],context),[.9,.1])
        with torch.no_grad():net[2].weight.normal_(std=.01)
        buffer=io.BytesIO();torch.save(context,buffer);buffer.seek(0)
        loaded=torch.load(buffer,weights_only=True)
        from jevmahjong.policy.kan_context import feature_vectors
        expected=net(torch.tensor(feature_vectors(req,VERSION),dtype=torch.float64)).flatten()
        self.assertTrue(torch.allclose(context_scores(req,loaded),expected,atol=1e-12,rtol=0))
        for bad in (257,True,0):
            with self.assertRaises(ValueError):make_network(bad,len(names))
            with self.assertRaises(ValueError):context_scores(req,dict(loaded,hidden=bad))

    def test_public_route_loss_keeps_negative_states_and_complete_menus(self):
        rows=[pair(100),pair(104),pair(108)]
        def reverse_credit(row):
            credit=row['training_credit']
            credit['control'],credit['branch']=credit['branch'],credit['control']
            row['training_advantage']=-row['training_advantage']
        reverse_credit(rows[1])
        cfg=dict(policy_state_weighting='public_routes',train_scopes=['calling'],
                 route_loss_weights={'closed_chi_tanyao_gain':4},max_example_weight=4.)
        classify=lambda gs:{'closed_chi_tanyao_gain':gs[:2]}
        with patch('jevmahjong.training.call_head.route_groups',side_effect=classify):
            groups,report=training_groups(rows,cfg,VERSION)
            flipped=copy.deepcopy(rows)
            for row in flipped:reverse_credit(row)
            other,_=training_groups(flipped,cfg,VERSION)
            with self.assertRaisesRegex(ValueError,'state cap'):
                training_groups(rows,dict(cfg,max_example_weight=1.),VERSION)
        self.assertEqual([g['weight'] for g in groups],[4/3,4/3,1/3])
        self.assertEqual([g['weight'] for g in other],[g['weight'] for g in groups])
        self.assertEqual([len(g['utility']) for g in groups],[2,2,2])
        self.assertEqual(len(groups),3)
        self.assertGreater(float(groups[0]['utility'][1]),0)
        self.assertLess(float(groups[1]['utility'][1]),0)
        self.assertFalse(report['outcome_sign_reweighting'])
        self.assertFalse(report['public_route_loss']['held_weighted'])

    def test_wins_select_even_with_lower_points_and_reject_risk_or_roundoff(self):
        base = selection()
        cases = [
            (selection(win=.02, fast=.01, utility=.0225, points=-500), True),
            (selection(win=-.01, fast=.2, utility=.04, points=10000), False),
            (selection(win=.03, risk=.02, utility=.01), False),
            (selection(win=.03, utility=-.01), False),
            (selection(win=float('nan'), utility=.1), False),
            (selection(win=1e-14, utility=.1), False),
        ]
        for p, expected in cases:
            with self.subTest(p=p):
                wrap = lambda q: dict(validation_selection=q)
                self.assertEqual(win_policy_improved(wrap(p), wrap(base), wrap(base), .01), expected)
                self.assertEqual(manual_win_improves(p, base, base, .01), expected)
        best = selection(win=.02, fast=.01, utility=.02)
        faster = selection(win=.02, fast=.02, utility=.025)
        self.assertTrue(manual_win_improves(faster, best, base, .01))
        faster[WIN_METRIC] = .019
        self.assertFalse(manual_win_improves(faster, best, base, .01))

    def test_full_menu_credit_tracks_win_risk_and_points_separately(self):
        g = pack_options([pair(100)], feature_version=VERSION)[0]
        self.assertEqual(g['credit'][1]['own_win'], .125)
        self.assertEqual(g['credit'][1]['net_points'], -500.)
        self.assertEqual(g['credit'][1]['dealin'], 0.)
        net = torch.nn.Linear(len(feature_names(VERSION)), 1, bias=False).double()
        with torch.no_grad():
            net.weight.zero_()
            net.weight[0, feature_names(VERSION).index('pon')] = .3
        prof = policy_profile(net, [g], [dict(path='test', seed_start=100, seeds=4)], .05)[0]
        combined = selection_profile([prof], 'all_sources')
        self.assertEqual(combined[WIN_METRIC], .125)
        self.assertEqual(combined['mean_sampled_net_point_gain_per_state'], -500.)
        self.assertAlmostEqual(combined['mean_sampled_fast_reward_change_per_state'], .05)
        actual = dict(prof)
        self.assertEqual(check_profiles(actual, prof), 0.)
        for key in [WIN_METRIC, 'mean_sampled_dealin_change_per_state']:
            wrong = dict(prof); wrong[key] += .1
            with self.assertRaises(ValueError):
                check_profiles(actual, wrong)

    def test_real_optimizer_and_independent_audit_allow_win_gain_with_lower_points(self):
        from jevmahjong.policy.call_tile_context import VERSION as TILE_VERSION
        for version in (VERSION,TILE_VERSION):
            with self.subTest(version=version):self._real_fit_and_audit(version)
        self._real_fit_and_audit(VERSION,kan=True)

    def test_same_component_continuation_preserves_start_and_passes_independent_audit(self):
        from jevmahjong.policy.call_tile_context import VERSION as TILE_VERSION
        self._real_fit_and_audit(TILE_VERSION,warm=True)

    def test_kan_tile_optimizer_and_independent_audit_preserve_the_win_objective(self):
        from jevmahjong.policy.kan_tile_context import VERSION as KAN_TILE_VERSION
        self._real_fit_and_audit(KAN_TILE_VERSION,kan=True)

    def _real_fit_and_audit(self,version,kan=False,warm=False):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roles = {}
            for role in ('discard', 'call'):
                directory = root / role; directory.mkdir(); torch.save({}, directory / 'head.pt')
                roles[role + '_checkpoint'] = str(directory)
            source = dict(path=str(root / 'collection/source_0'), seed_start=100, seeds=4, walls=4)
            parts = dict(train=[pair(100)], held=[pair(103)])
            if kan:
                for rows in parts.values():
                    for row in rows:row.update(scope='ordinary',public_request=request('daiminkan'))
            cfg = dict(**roles, out=str(root / 'fit'), model_id='toy-win-context',
                       fit_model='residual_policy', train_scopes=['calling'], decision_filter='all',
                       data=[source['path']], reward=dict(mode='own_win_first_v1'),
                       feature_version=version, selection_validation='all_sources',
                       selection_objective='hand_win', max_held_dealin_increase=.01,
                       reference_prior_strength=.05, policy_state_weighting='public_routes',
                       route_loss_weights={'closed_pon_yakuhai':4},
                       max_example_weight=4., hidden=4, seed=17, lr=.02, weight_decay=.001,
                       batch=1, epochs=12, patience_epochs=3)
            if kan:cfg.update(train_scopes=['ordinary'],decision_filter='kan_windows',policy_state_weighting='uniform')
            if warm:
                from jevmahjong.training.context_network import make_network
                from jevmahjong.policy.call_baseline import transform
                torch.manual_seed(73);net=make_network(4,len(feature_names(version)))
                with torch.no_grad():net[2].weight.fill_(.001)
                context=dict(version=version,names=list(feature_names(version)),model_kind='mlp',
                    hidden=4,state_dict=copy.deepcopy(net.state_dict()),reference_prior_strength=.05)
                path=root/'baseline.pt';torch.save(dict(kan_context=context),path)
                cfg.update(context_initialization='calling_baseline',
                    call_context_baseline=dict(checkpoint=str(path),sha256=digest(path)))
                self.enterContext(patch('jevmahjong.training.context_initialization.initial_context',return_value=context))
                self.enterContext(patch('jevmahjong.policy.call_baseline.load',return_value=context))
                for rows in parts.values():
                    for row in rows:
                        row['original_reference']=FrozenHead()(row['decide'],row['options']).detach().softmax(-1)
                        row['reference']=transform(row['public_request'],row['original_reference'],context)
            fitfile = root / 'fit-config.json'; fitfile.write_text(json.dumps(cfg))

            def prepare(_):
                out = Path(cfg['out']); out.mkdir()
                protocol = dict(config=cfg, model_files={}, code_sha256={})
                if warm:protocol['call_context_baseline']=cfg['call_context_baseline']
                (out / 'protocol.json').write_text(json.dumps(protocol))
                return out, protocol, parts, {}, [source], {k: FrozenHead() for k in ('ordinary', 'calling')}

            with patch('jevmahjong.training.context_policy.prepare', side_effect=prepare):
                fit(cfg)
            result = json.loads((root / 'fit/result.json').read_text())
            if warm:
                first=torch.load(result['curve'][0]['checkpoint'],weights_only=True)
                self.assertTrue(first['context_replaces_calling_baseline'])
                self.assertTrue(torch.count_nonzero(first['kan_context']['state_dict']['2.weight']))
                self.assertTrue(all(torch.equal(v,context['state_dict'][k])
                    for k,v in first['kan_context']['state_dict'].items()))
            self.assertGreater(result['selected']['epoch'], 0)
            self.assertEqual(result['selected']['validation_selection'][WIN_METRIC], .125)
            self.assertEqual(result['selected']['validation_selection']['mean_sampled_net_point_gain_per_state'], -500.)
            data_audit = root / 'data-audit'; data_audit.mkdir()
            (data_audit / 'result.json').write_text(json.dumps(dict(status='passed_complete')))
            (data_audit / 'protocol.json').write_text(json.dumps(dict(
                config=dict(collection=str(root / 'collection')), model_files={}, code_sha256={}, source_files={})))
            checkcfg = dict(**roles, out=str(root / 'audit'), fit_config=str(fitfile),
                            fit_config_sha256=digest(fitfile), data_audits=[str(data_audit)],
                            expected=dict(wall_groups=4, features=len(feature_names(version)),
                                          parameters=4*(len(feature_names(version))+2),
                                          policy_state_weighting=cfg['policy_state_weighting']))
            snapshot = lambda c: dict(config=c, model_files={}, code_sha256={})
            with patch('jevmahjong.evaluation.context_audit.load_sources', return_value=(parts, {}, [source])), \
                 patch('jevmahjong.evaluation.context_audit.make_head', side_effect=lambda *a: FrozenHead()), \
                 patch('jevmahjong.evaluation.context_audit.snapshot', side_effect=snapshot):
                checked = audit(checkcfg)
                self.assertTrue(checked['eligible_for_development'])
                self.assertEqual(checked['selection_objective'], 'hand_win')
                self.assertLess(checked['selected']['mean_sampled_net_point_gain_per_state'], 0)
                if warm:self.assertTrue(checked['context_replaces_calling_baseline'])
                if kan:
                    self.assertEqual(checked['activation'],'kan_windows')
                    self.assertEqual(checked['held_kan_routes']['daiminkan'][WIN_METRIC],.125)
                else:
                    self.assertEqual(checked['held_routes']['closed_pon_yakuhai']['sampled_positive_call_states'], 1)
                    self.assertEqual(checked['held_routes']['closed_pon_yakuhai']['recovered_sample_positive_calls'], 1)
                missing = dict(checkcfg, out=str(root / 'missing-data-audit'), data_audits=[])
                with self.assertRaisesRegex(ValueError, 'Every win-call'):
                    audit(missing)
                # Corrupt all saved win summaries consistently. The independent
                # recomputation must reject them, even when selection metadata agrees.
                wrong = copy.deepcopy(result)
                for item in wrong['curve']:
                    item['validation_selection'][WIN_METRIC] += .1
                    for prof in item['validation_sources']:
                        prof[WIN_METRIC] += .1
                wrong['selected'] = wrong['curve'][result['selected']['epoch']]
                (root / 'fit/result.json').write_text(json.dumps(wrong))
                with self.assertRaisesRegex(ValueError, 'Independent held profile'):
                    audit(dict(checkcfg, out=str(root / 'corrupt-audit')))
                if warm:
                    changed=copy.deepcopy(result);first=changed['curve'][0];path=Path(first['checkpoint'])
                    model=torch.load(path,weights_only=True);model['kan_context']['state_dict']['0.weight'][0,0]+=.01
                    torch.save(model,path);first['sha256']=digest(path)
                    (root/'fit/result.json').write_text(json.dumps(changed))
                    with self.assertRaisesRegex(ValueError,'Starting parameters differ from fixed calling baseline'):
                        audit(dict(checkcfg,out=str(root/'wrong-continuation-start')))


if __name__ == '__main__':
    unittest.main()
