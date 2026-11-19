import copy
import json
from pathlib import Path
import unittest

from tools.continue_endgame_fourth_lora import validate_config,research_screen,progress_state,validate_resume_state,continuation_coverage
from tools.train_endgame_fourth_lora_balanced import balanced_schedule


def config():
    return json.loads((Path(__file__).resolve().parents[2]/'configs/training/endgame-fourth-lora-extension-v1.json').read_text())


def metric(top2=.24):
    return dict(fourth=dict(top2=top2,fourth=.8,soft_top2=.19),legacy=dict(
        third_pursuit=dict(candidate=.45,original_policy=.43),utility_by_stratum=dict(third=.11,hold=.97),original_expert_ce=1.1),
        native_third_kl=dict(mean=.5))


def anchor():
    return dict(formal=metric(.22),best_audited_parent_top2=.23,parent_best_live_soft=.19)


class FourthLoRAExtensionTests(unittest.TestCase):
    def test_kl_or_fourth_tail_cannot_veto_research_but_not_release_qualified(self):
        result=research_screen(160,metric(),anchor())
        self.assertTrue(result['eligible_for_consumed_research']);self.assertFalse(result['release_qualified'])
        self.assertFalse(research_screen(128,metric(),anchor())['eligible_for_consumed_research'])

    def test_third_holding_and_best_parent_are_still_required(self):
        for kind in ('third','hold','parent'):
            m=metric()
            if kind=='third':m['legacy']['third_pursuit']['candidate']=.44
            elif kind=='hold':m['legacy']['utility_by_stratum']['hold']=.94
            else:m['fourth']['top2']=.23
            self.assertFalse(research_screen(160,m,anchor())['eligible_for_consumed_research'])

    def test_observation_then_three_checks_not_two_updates(self):
        c=config();a=anchor();m=metric(.22);s=progress_state(128,m,{},None,a,c)
        for step in (160,192,224,256):
            s=progress_state(step,m,research_screen(step,m,a),s,a,c);self.assertFalse(s['stop'])
        self.assertTrue(progress_state(288,m,research_screen(288,m,a),s,a,c)['stop'])

    def test_soft_progress_is_not_research_qualification(self):
        c=config();a=anchor();m=metric(.22);s=progress_state(128,m,{},None,a,c)
        for step in range(160,385,32):
            m['fourth']['soft_top2']+=.001
            s=progress_state(step,m,research_screen(step,m,a),s,a,c)
            self.assertFalse(s['stop']);self.assertIsNone(s['selected_research_step'])

    def test_optimizer_state_must_be_128_with_original_lrs_and_full_exposure(self):
        ids=list(range(826));s=dict(step=128,seen_ids=ids,torch_rng=[],cuda_rng=[],optimizer=dict(
            param_groups=[dict(lr=2e-6,weight_decay=.001),dict(lr=1e-5,weight_decay=.001)],state={0:dict(step=128)}))
        validate_resume_state(s,ids)
        for change in ('step','lr','moment'):
            bad=copy.deepcopy(s)
            if change=='step':bad['step']=0
            elif change=='lr':bad['optimizer']['param_groups'][0]['lr']=1e-6
            else:bad['optimizer']['state'][0]['step']=0
            with self.assertRaises(ValueError):validate_resume_state(bad,ids)

    def test_sampler_prefix_does_not_depend_on_new_total_length(self):
        from tools.train_endgame_fourth_lora_balanced import QUOTAS
        rows=[{'questions':{'action':{'src':k}}} for k in QUOTAS for _ in range(41)]
        self.assertEqual(balanced_schedule(rows,128),balanced_schedule(rows,384)[:128])

    def test_continuation_keeps_prior_exposure_without_forcing_a_sampler_restart(self):
        schedule=[list(range(826))]+[[0] for _ in range(191)]
        result=continuation_coverage(schedule)
        self.assertEqual(result,{'cumulative_unique':826,'extension_unique':1})
        with self.assertRaises(ValueError):continuation_coverage([[0] for _ in range(192)])

    def test_config_disallows_restart_and_unregistered_data_change(self):
        c=config();validate_config(c)
        for changes in ({'resume_optimizer_and_rng':False},{'repeat_warmup':True},
                        {'new_training_records':100},{'max_total_steps':4096}):
            with self.assertRaises(ValueError):validate_config(dict(c,**changes))


if __name__=='__main__':unittest.main()
