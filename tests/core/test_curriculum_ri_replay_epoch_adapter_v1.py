import copy
import random
from types import SimpleNamespace
import unittest
from tools.curriculum_ri_replay_epoch_adapter_v1 import EpochReplay,random_proxy
from tools.curriculum_ri_every_update_replay_v1 import build,MASS,RI
from tools.curriculum_ri_inverse_repeat_loss_v1 import FIELD,IDENTIFIER
from tools import curriculum_progression_parent_loss_v1 as retained
from tests.core.test_curriculum_ri_inverse_repeat_loss_v1 import example


class EpochAdapterTests(unittest.TestCase):
    def fixture(self):
        rows=[];categories=[]
        for source in MASS:
            for _ in range(RI.index(source)+1 if source in RI else 2):
                raw,_=example(source);raw['_meta']={'id':str(len(rows)),'group_id':str(len(rows))}
                rows.append(raw);categories.append(source)
        plans=[build(categories,list(range(len(rows))),seed,batch=9) for seed in (7,8)]
        adapter=EpochReplay(rows,plans)
        args=SimpleNamespace(batch=1,accum=9,rlcr=False,p_none=0.,p_none_distract=0.,p_distract=0.,p_none_pair=0.)
        return rows,plans,adapter,args

    def encode(self,model,tok,args,resolved,epoch):
        result=[]
        for raw in resolved:
            q=raw['questions']['action'];keys=list(q['criteria'])
            rec={'questions':[dict(qid='action',src=q['src'],keys=keys,label=keys.index(q['label']),
                target=[q['target'].get(k,0.) for k in keys] if q.get('target') is not None else None)]}
            retained.annotate(rec,raw)
            result.append(SimpleNamespace(rec=rec,enc={'ids':[int(raw['_meta']['id'])]}))
        return result

    def test_both_epochs_resolve_their_own_originals_and_weights(self):
        rows,plans,a,args=self.fixture();original=copy.deepcopy(rows);order=list(range(a.size))
        for epoch in (0,1):
            a.shuffle(order)
            for position in order:
                v=a.encode_batch(self.encode,None,None,args,[a.virtual_requests[position]],epoch)[0]
                i=plans[epoch]['indices'][position];q=v.rec['questions'][0]
                self.assertEqual(q[IDENTIFIER],rows[i]['_meta']['id'])
                self.assertEqual(q[FIELD],plans[epoch]['weights_by_original_index'][str(i)])
                self.assertEqual(v.enc,{'ids':[i]})
        self.assertEqual(a.finish()['physical_exposures'],2*a.size)
        self.assertEqual(rows,original)

    def test_unique_admission_must_succeed_before_expansion(self):
        rows,plans,a,args=self.fixture();n=len(rows)
        report=dict(requested_records=n,admitted_records=n,rejected_records=0,rejected=[])
        pool,expanded=a.admit(rows,report)
        self.assertEqual(len(pool),a.size);self.assertEqual(expanded['unique_original_records'],n)
        with self.assertRaises(ValueError):a.admit(rows[:-1],report)
        with self.assertRaises(ValueError):a.admit(rows,dict(report,rejected_records=1))

    def test_stale_epochs_skips_early_shuffle_and_augmentation_rejected(self):
        rows,plans,a,args=self.fixture();order=list(range(a.size));a.shuffle(order)
        with self.assertRaises(ValueError):a.shuffle(order)
        with self.assertRaises(ValueError):a.encode_batch(self.encode,None,None,args,[a.virtual_requests[1]],0)
        with self.assertRaises(ValueError):a.encode_batch(self.encode,None,None,args,[a.virtual_requests[0]],1)
        args.p_none=.1
        with self.assertRaises(ValueError):a.encode_batch(self.encode,None,None,args,[a.virtual_requests[0]],0)
        with self.assertRaises(ValueError):a.finish()

    def test_only_global_rng_is_intercepted_and_corrupt_mass_rejected(self):
        rows,plans,a,args=self.fixture();proxy,created=random_proxy(random,123,a)
        self.assertEqual(proxy.Random(99).random(),random.Random(99).random())
        proxy.Random(123).shuffle(list(range(a.size)));self.assertEqual(len(created),1)
        with self.assertRaises(ValueError):proxy.Random(123)
        broken=copy.deepcopy(plans);broken[0]['weights_by_original_index']['0']*=2
        with self.assertRaises(ValueError):EpochReplay(rows,broken)


if __name__=='__main__':unittest.main()
