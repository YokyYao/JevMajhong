import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
from collections import Counter
from jevmahjong.core.io import digest
from jevmahjong.data.base_lora_mix import choose,kan_route,prepare


class BaseLoraMixTests(unittest.TestCase):
    def test_bounded_repetition_is_deterministic_and_never_filters_by_outcome(self):
        rows=[('positive','p'),('negative','n')]
        selected=choose(rows,5,3,19)
        self.assertEqual(selected,choose(rows,5,3,19))
        self.assertEqual({p for p,_ in selected},{'positive','negative'})
        self.assertLessEqual(max(Counter(p for p,_ in selected).values()),3)
        with self.assertRaisesRegex(ValueError,'coverage'):choose(rows,7,3,19)

    def test_kan_stratum_uses_available_operation_not_the_pseudo_label(self):
        record=dict(questions={'action':dict(criteria={'a0':{'type':'kakan'},'a1':{'type':'dahai'}},label='a1')})
        self.assertEqual(kan_route(record),'kakan')

    def test_materialized_files_keep_whole_groups_separate_and_preserve_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);pool=root/'pool';pool.mkdir();db=sqlite3.connect(pool/'pool-private.sqlite')
            db.execute('CREATE TABLE cases (position TEXT,split TEXT,group_key TEXT,phase TEXT,label_type TEXT,choice TEXT,view TEXT,priority INTEGER,payload TEXT)')
            for split,group in [('train','train-match'),('validation','held-match')]:
                r=dict(state={'public':'only'},questions={'action':dict(criteria={'a0':{'type':'dahai'}},label='a0')},_meta=dict(id=group,group_id=group,phase='closed_discard',view='native'))
                db.execute('INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?)',(group,split,group,'closed_discard','expert_replay','dahai','native',1,json.dumps(r)))
            db.commit();db.close();(pool/'result.json').write_text(json.dumps(dict(status='raw_request_pool_ready_not_yet_balanced',requires_masked_soft_target_loss=True)))
            spec=dict(phase='closed_discard',label_type='expert_replay',count=1,max_repeats=1)
            cfg=dict(pool=str(pool),pool_sha256=digest(pool/'pool-private.sqlite'),pool_result_sha256=digest(pool/'result.json'),out=str(root/'data'),seed=11,mixture={'train':[spec],'validation':[spec]})
            result=prepare(cfg)
            self.assertEqual(result['train_validation_group_overlap'],0)
            self.assertEqual(result['files']['train']['records'],1)
            self.assertEqual(result['files']['validation']['unique_positions'],1)
            self.assertTrue(result['unknown_actions_are_not_negative_labels'])


if __name__=='__main__':unittest.main()
