import copy
import json
import sqlite3
import unittest
from jevmahjong.data.extend_pool import append_records,canonical_groups
from jevmahjong.data.base_lora_pool import normalized_request,position


def record(group,tile='1m'):
    req=dict(state=dict(seat=0,hand=tile,rivers=['']*4,melds=[[]]*4),
        questions={'action':dict(criteria={'a0':dict(type='dahai',pai=tile)},label='a0',target={'a0':1.},src='paired_future_masked')})
    return normalized_request(req,group,'paired_future_masked',0)


class ExtendPoolTests(unittest.TestCase):
    def test_pre_and_post_reach_share_one_position_but_never_average_different_menus(self):
        before=record('native:100');before['questions']['action']['criteria']['a1']=dict(type='reach')
        after=record('native:100')
        early=[dict(episode_key=(100,1,24,6),public_request=before,training_advantage=-100.)]
        late=[dict(episode_key=(100,1,25,1),public_request=after,training_advantage=100.)]
        untouched=copy.deepcopy([late,early])
        self.assertEqual(len(canonical_groups([late,early])[0]),2)
        kept,omitted=canonical_groups([late,early],'earliest_observed_decision_v1')
        self.assertEqual(kept,[early]);self.assertEqual(omitted[0]['omitted_origin'],[100,1,25])
        self.assertFalse(omitted[0]['legal_menus_equal']);self.assertEqual([late,early],untouched)
        changed=copy.deepcopy(late);changed[0]['public_request']['state']['scores']=[26000]*4
        with self.assertRaisesRegex(ValueError,'distinct public states'):
            canonical_groups([early,changed],'earliest_observed_decision_v1')
        with self.assertRaisesRegex(ValueError,'Unknown'):canonical_groups([early],'best_reward')

    def db(self):
        db=sqlite3.connect(':memory:');db.execute('CREATE TABLE cases (position TEXT,split TEXT,group_key TEXT,phase TEXT,label_type TEXT,choice TEXT,view TEXT,priority INTEGER,payload TEXT,PRIMARY KEY(position,split))')
        append_records(db,[('train',record('native:1')),('validation',record('native:4'))],set(),1)
        return db

    def test_unchanged_parent_positions_and_validation_survive_novel_additions(self):
        db=self.db();before=db.execute('SELECT * FROM cases').fetchall()
        counts=append_records(db,[('train',record('native:2')),('validation',record('native:8'))],set(),2)
        self.assertEqual(dict(counts),dict(train=1,validation=1))
        self.assertTrue(set(before)<=set(db.execute('SELECT * FROM cases').fetchall()));db.close()

    def test_historical_holdouts_repeats_and_cross_source_group_leaks_are_rejected(self):
        db=self.db();counts=append_records(db,[('train',record('native:4','2m')),('train',record('native:12'))],{'native:12'},2)
        self.assertEqual(counts['historically_held_train_rows_blocked'],2)
        with self.assertRaisesRegex(ValueError,'parent training'):append_records(db,[('validation',record('native:1','2m'))],set(),2)
        with self.assertRaisesRegex(ValueError,'physical position'):append_records(db,[('train',record('native:1'))],set(),2)
        with self.assertRaisesRegex(ValueError,'leak whole groups'):
            append_records(db,[('train',record('native:9')),('validation',record('native:9','2m'))],set(),2)
        db.close()


if __name__=='__main__':unittest.main()
