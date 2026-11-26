import copy
import random
import unittest
from types import SimpleNamespace
from tools import curriculum_ri_parent_anchor_v1 as anchor
from tools import curriculum_ri_parent_anchor_epoch_v2 as epoch
from tools import curriculum_ri_coverage_replay_v2 as schedule
from tools import curriculum_progression_parent_loss_v1 as retained
from tests.core import test_curriculum_ri_coverage_replay_v2 as schedule_fixtures
from tests.core import test_curriculum_ri_parent_anchor_v1 as fixtures


class PostEncodingAnchorTests(unittest.TestCase):
    def population(self):
        categories,baseline,_=schedule_fixtures.CoverageReplayTests().fixture()
        rows=[];predictions=[]
        for i,source in enumerate(categories):
            kinds=['dahai','reach','dahai'] if source in anchor.RI_SOURCES else ['dahai']*3
            label=1 if source.endswith(':riichi_yes') else 2
            target=[.3,0.,.7] if ':mc:' in source else None
            row,_=fixtures.fixture(source,kinds,label,str(i),target);row['state']={'public_test_index':i};rows.append(row)
            if source in anchor.RI_SOURCES:predictions.append(dict(id=str(i),keys=['k0','k1','k2'],p=[.15,.6,.25]))
        plans=[schedule.build(categories,baseline,s,batch=11) for s in (7,8)]
        return rows,plans,anchor.make_targets(rows,predictions)

    def args(self):
        return SimpleNamespace(batch=1,accum=11,rlcr=False,p_none=0.,p_none_distract=0.,p_distract=0.,p_none_pair=0.)

    def encoder(self,model,tok,args,rows,number):
        result=[]
        for row in rows:
            # Match the relevant upstream contract: augmentation removes _meta.
            augmented={k:copy.deepcopy(v) for k,v in row.items() if k!='_meta'}
            raw=augmented['questions']['action'];keys=list(raw['criteria'])
            question=dict(qid='action',src=raw['src'],keys=keys,label=keys.index(raw['label']),
                          target=[raw['target'].get(k,0.) for k in keys] if raw.get('target') else None)
            rec=retained.annotate(dict(questions=[question]),augmented)
            self.assertNotIn(anchor.FIELD,question)
            self.assertNotIn(anchor.MARKER,question)
            encoding={'ids':[row['state']['public_test_index'],9]}
            result.append(SimpleNamespace(rec=rec,enc=encoding))
        return result

    def test_metadata_stripped_encoder_supports_all_full_epoch_positions(self):
        rows,plans,targets=self.population();before=copy.deepcopy(rows)
        replay=epoch.AnchoredEpochReplay(rows,plans,targets);replay.shuffle(list(range(replay.size)))
        for position,virtual in enumerate(replay.virtual_requests):
            variant=replay.encode_batch(self.encoder,None,None,self.args(),[virtual],0)[0]
            index=plans[0]['indices'][position];q=variant.rec['questions'][0]
            self.assertEqual(variant.enc,{'ids':[index,9]})
            self.assertEqual(q[anchor.MARKER],anchor.METHOD)
            self.assertEqual(q[anchor.FIELD] is not None,q['src'] in anchor.RI_SOURCES)
        self.assertEqual(rows,before)

    def test_missing_targets_and_repeated_attachment_rejected(self):
        rows,plans,targets=self.population();missing=dict(targets);missing.pop(next(iter(missing)))
        with self.assertRaises(ValueError):epoch.AnchoredEpochReplay(rows,plans,missing)
        replay=epoch.AnchoredEpochReplay(rows,plans,targets);replay.shuffle(list(range(replay.size)))
        variant=replay.encode_batch(self.encoder,None,None,self.args(),[replay.virtual_requests[0]],0)[0]
        original=rows[plans[0]['indices'][0]]
        with self.assertRaises(ValueError):epoch.attach(variant.rec,original,targets)

    def test_target_identity_is_not_recovered_from_encoded_metadata(self):
        rows,plans,targets=self.population();replay=epoch.AnchoredEpochReplay(rows,plans,targets)
        replay.shuffle(list(range(replay.size)))
        for position in range(11):
            variant=replay.encode_batch(self.encoder,None,None,self.args(),[replay.virtual_requests[position]],0)[0]
            self.assertNotIn('_meta',variant.rec)
            q=variant.rec['questions'][0]
            if q['src'] in anchor.RI_SOURCES:self.assertEqual(q[anchor.FIELD]['probability'],.6)


if __name__=='__main__':unittest.main()
