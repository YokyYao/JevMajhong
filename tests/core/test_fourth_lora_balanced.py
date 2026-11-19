from collections import Counter
import json
from pathlib import Path
import unittest

from tools.train_endgame_fourth_lora_balanced import balanced_schedule,record_loss_coefficient,validate_config,QUOTAS


class FourthBalancedLoRATests(unittest.TestCase):
    def test_every_step_has_every_source_and_full_coverage_before_observation_ends(self):
        counts={'replay_reference':544,'replay_reference_leading':64,'endgame_terminal_win_fourth_chase':94,
            'endgame_terminal_win_third':17,'endgame_terminal_third':29,'endgame_terminal_hold':23,
            'endgame_terminal_fourth_chase_improve':16,'endgame_terminal_fourth_chase_keep':39}
        rows=[{'questions':{'action':{'src':k}}} for k,n in counts.items() for _ in range(n)]
        batches=balanced_schedule(rows,128)
        self.assertEqual(batches,balanced_schedule(rows,128))
        for batch in batches:
            self.assertEqual(len(batch),32)
            self.assertEqual(dict(Counter(rows[i]['questions']['action']['src'] for i in batch)),QUOTAS)
        self.assertEqual(len({i for b in batches[:39] for i in b}),826)
        for source,n in counts.items():
            draws=[i for b in batches for i in b if rows[i]['questions']['action']['src']==source]
            self.assertEqual(len(set(draws[:n])),n)

    def test_stratification_preserves_source_loss_mass(self):
        masses={k:1/8 for k in QUOTAS}
        self.assertAlmostEqual(sum(QUOTAS[k]*record_loss_coefficient(k,masses) for k in QUOTAS),1.)
        for k in QUOTAS:self.assertAlmostEqual(QUOTAS[k]*record_loss_coefficient(k,masses),masses[k])
        with self.assertRaises(ValueError):record_loss_coefficient('x',{'x':1.})

    def test_config_retains_existing_lora_and_risk_preference(self):
        cfg=json.loads((Path(__file__).resolve().parents[2]/'configs/training/endgame-fourth-lora-balanced-v1.json').read_text())
        validate_config(cfg)
        for changes in ({'fourth_escape_constraint':'veto'},{'early_stop_observation_start':0},
                        {'trainable_scope':'base_weights'},{'lora_rank':32}):
            with self.assertRaises(ValueError):validate_config(dict(cfg,**changes))


if __name__=='__main__':unittest.main()
