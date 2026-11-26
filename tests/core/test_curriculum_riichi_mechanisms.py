import copy
import math
import unittest

from tools.curriculum_basic_repair import MASS,weights
from tools.curriculum_riichi_mechanisms import score_rows,distribution,update_mass,summarize_mass


class RiichiMechanismTests(unittest.TestCase):
    def fixture(self):
        rows=[];predictions=[]
        values={'legacy:riichi_yes':[.35,.25,.4],'legacy:riichi_no':[.45,.25,.3],
            'teacher:riichi_yes':[.1,.1,.8],'teacher:riichi_no':[.2,.1,.7]}
        for source in MASS:
            plain=source.endswith(':plain');yes=source.endswith(':riichi_yes')
            criteria={'a':{'type':'dahai'},'b':{'type':'dahai'}}
            if not plain:criteria['r']={'type':'reach'}
            q={'src':'curriculum1:'+source,'criteria':criteria,'label':'r' if yes else 'b'}
            rows.append({'_meta':{'id':source,'group_id':source},'questions':{'action':q}})
            predictions.append({'id':source,'keys':list(criteria),'p':[.6,.4] if plain else values[source]})
        return rows,predictions

    def test_discard_dispersion_is_counted_without_changing_action_choice(self):
        rows,predictions=self.fixture();summary,cases=score_rows(rows,predictions)
        self.assertEqual(summary['pooled']['windows'],4)
        self.assertEqual(summary['pooled']['class_probability_auc'],.75)
        self.assertEqual(summary['sources']['legacy']['confusion']['raw_argmax'],{'tp':1,'fn':0,'tn':1,'fp':0})
        self.assertEqual(summary['sources']['legacy']['confusion']['fixed_full_class_0_5'],{'tp':0,'fn':1,'tn':1,'fp':0})
        self.assertEqual(summary['pooled']['fixed_decoder_disagreements']['raw_reach_only'],1)
        self.assertEqual(summary['pooled']['fixed_decoder_disagreements']['teacher_yes'],1)
        self.assertTrue(all(r['id'] not in ('legacy:plain','teacher:plain') for r in cases))

    def test_exact_ce_decomposition_and_class_brier_on_known_probabilities(self):
        rows,predictions=self.fixture();summary,cases=score_rows(rows,predictions)
        for r in cases:
            self.assertAlmostEqual(r['class_nll']+r['conditional_discard_nll'],r['normalized_action_nll'])
            if r['truth']:self.assertEqual(r['conditional_discard_nll'],0)
        expected=(.6**2+.3**2+.2**2+.7**2)/4
        self.assertAlmostEqual(summary['pooled']['class_brier'],expected)
        self.assertLess(summary['pooled']['max_nll_decomposition_error'],1e-12)
        self.assertEqual(sum(b['n'] for b in summary['pooled']['equal_width_calibration_bins']),4)

    def test_candidate_ties_preserve_original_order_and_fixed_threshold(self):
        rows,predictions=self.fixture();predictions[0]['p']=[.5,.5]
        index=next(i for i,p in enumerate(predictions) if p['id']=='legacy:riichi_yes')
        predictions[index]['p']=[.5,0,.5]
        _,cases=score_rows(rows,predictions)
        r=next(r for r in cases if r['id']=='legacy:riichi_yes')
        self.assertFalse(r['raw_argmax']);self.assertTrue(r['fixed_full_class_0_5'])
        self.assertEqual(r['logit_margin_vs_best_discard'],0)

    def test_invalid_tags_duplicate_ids_zero_mass_and_menus_fail(self):
        for change in ('tag','duplicate','zero','menu','missing'):
            rows,predictions=self.fixture()
            index=next(i for i,r in enumerate(rows) if r['_meta']['id']=='legacy:riichi_yes')
            if change=='tag':rows[index]['questions']['action']['label']='a'
            elif change=='duplicate':predictions[-1]=copy.deepcopy(predictions[0])
            elif change=='zero':predictions[index]['p']=[.5,.5,0]
            elif change=='menu':rows[index]['questions']['action']['criteria']['b']['type']='pon'
            else:predictions.pop()
            with self.assertRaises(ValueError,msg=change):score_rows(rows,predictions)

    def test_actual_weight_mass_can_vary_by_update_but_balance_by_epoch(self):
        counts=dict.fromkeys(MASS,10);scaling=weights(counts);updates=[]
        for step in (1,2):
            n={k:5 for k in MASS}
            for k in n:
                if k.endswith(':riichi_yes'):n[k]=9 if step==1 else 1
                if k.endswith(':riichi_no'):n[k]=1 if step==1 else 9
            loss=sum(n[k]*scaling[k] for k in n)/sum(n.values())
            updates.append(dict(local_step=step,source_counts=n,source_ce=dict.fromkeys(MASS,1.),loss=loss))
        actual=update_mass(updates,scaling,counts);summary=summarize_mass(actual)
        self.assertAlmostEqual(summary['aggregate_riichi_positive_fraction'],.5)
        self.assertEqual(summary['steps_positive_fraction_above_0_75'],1)
        self.assertEqual(summary['steps_positive_fraction_below_0_25'],1)
        self.assertEqual(summary['maximum_loss_reconstruction_error'],0)
        updates[0]['loss']+=.1
        with self.assertRaises(ValueError):update_mass(updates,scaling,counts)

    def test_diagnostics_do_not_tolerate_missing_sources_or_changed_weights(self):
        counts=dict.fromkeys(MASS,10);scaling=weights(counts);scaling[next(iter(scaling))]*=2
        with self.assertRaises(ValueError):update_mass([],scaling,counts)
        with self.assertRaises(ValueError):distribution([])
        self.assertEqual(distribution([1,3])['median'],2)
        self.assertAlmostEqual(distribution([1,3])['p10'],1.2)


if __name__=='__main__':unittest.main()
