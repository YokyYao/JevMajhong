import json
from pathlib import Path
import tempfile
import unittest
from jevmahjong.training.metric_journal import MetricJournal,aggregate_complete_matches,export_csv

class MetricsTests(unittest.TestCase):
    def test_match_weighting_is_not_pooled_hands(self):
        rows=[dict(completed=True,hands=2,wins=1,dealin_hands=0,rank=1,net_points=8000,seed=1),
              dict(completed=True,hands=8,wins=1,dealin_hands=2,rank=4,net_points=-4000,seed=1)]
        m=aggregate_complete_matches(rows)
        self.assertEqual(m['hand_win_rate'],.3125)
        self.assertEqual(m['seed_groups'],1)
        self.assertEqual(m['top2_rate'],.5)
        self.assertEqual(m['reward'],.25)

    def test_no_synthetic_sft_reward_or_unidentified_matches(self):
        with tempfile.TemporaryDirectory() as temp:
            journal=MetricJournal(temp,'run','stage1',100)
            with self.assertRaises(ValueError):journal.append('train',1,loss=.2,reward=.8)
            with self.assertRaises(ValueError):journal.append('evaluation',1,top2_rate=.5)
            event=journal.append('train',1,loss=.2,reward=None)
            self.assertEqual(event['global_step'],101)
            export_csv(temp)
            self.assertTrue((Path(temp)/'train_metrics.csv').exists())
            self.assertIsNone(json.loads((Path(temp)/'train_metrics.jsonl').read_text())['reward'])

    def test_reject_nonfinite_and_percent_mislabelled_as_fraction(self):
        with tempfile.TemporaryDirectory() as temp:
            j=MetricJournal(temp,'run','stage1')
            with self.assertRaises(ValueError):j.append('train',1,loss=float('nan'))
            with self.assertRaises(ValueError):
                j.append('evaluation',1,matches=16,seed_groups=2,checkpoint_sha256='abc',top2_rate=50)
