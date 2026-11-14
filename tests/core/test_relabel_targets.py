import copy
import json
from pathlib import Path
import tempfile
import unittest
from jevmahjong.core.io import digest
from jevmahjong.data.base_lora_pool import position
from jevmahjong.data.relabel_targets import materialize


class RelabelTargetsTests(unittest.TestCase):
    def fixture(self,root):
        def record(name,label_type):
            q=dict(criteria=dict(a0={'type':'dahai','pai':'1m'},a1={'type':'dahai','pai':'9p'},a2={'type':'reach'}),label='a0')
            if label_type=='paired_future_masked':q.update(target=dict(a0=.5,a1=.5,a2=0.),src='paired_future_masked')
            return dict(state=dict(seat=0,hand=[name],rivers=[[],[],[],[]],melds=[[],[],[],[]]),questions={'action':q},
                _meta=dict(id=name,group_id=name,phase='closed_discard',label_type=label_type))
        expert=record('expert','expert_replay');mc=record('mc','paired_future_masked');val=record('held','expert_replay')
        train=root/'original-train.jsonl';validation=root/'original-validation.jsonl'
        expert_bytes=(json.dumps(expert,separators=(',',':'))+'\n').encode()
        train.write_bytes(expert_bytes+(json.dumps(mc)+'\n').encode());validation.write_bytes((json.dumps(val,separators=(',',':'))+'\n').encode())
        parent={'files':{}}
        for split,path,n in [('train',train,2),('validation',validation,1)]:
            parent['files'][split]=dict(path=str(path),sha256=digest(path),records=n,unique_positions=n,groups=n)
        out=root/'new';out.mkdir()
        return parent,out,expert,mc,expert_bytes

    def test_preserves_order_expert_bytes_and_entire_validation(self):
        with tempfile.TemporaryDirectory() as temp:
            parent,out,expert,mc,expert_bytes=self.fixture(Path(temp));original=copy.deepcopy(mc)
            target=dict(a0=.05,a1=.95,a2=0.);files=materialize(parent,{position(mc,'mc'):target},out)
            self.assertTrue((out/'train.jsonl').read_bytes().startswith(expert_bytes))
            rows=[json.loads(x) for x in (out/'train.jsonl').read_text().splitlines()]
            self.assertEqual([r['_meta']['id'] for r in rows],['expert','mc'])
            self.assertEqual(rows[0],expert);self.assertEqual(rows[1]['questions']['action']['target'],target)
            self.assertEqual(rows[1]['questions']['action']['label'],'a1')
            rows[1]['questions']['action'].update(target=original['questions']['action']['target'],label='a0')
            self.assertEqual(rows[1],original)
            self.assertEqual((out/'validation.jsonl').read_bytes(),Path(parent['files']['validation']['path']).read_bytes())
            self.assertEqual(files['validation']['sha256'],parent['files']['validation']['sha256'])

    def test_rejects_expert_revision_or_unmatched_position(self):
        for mode in ['expert','missing']:
            with tempfile.TemporaryDirectory() as temp:
                parent,out,expert,mc,_=self.fixture(Path(temp))
                key=position(expert,'expert') if mode=='expert' else 'missing'
                with self.assertRaisesRegex(ValueError,'MC training position|omitted'):
                    materialize(parent,{key:dict(a0=.1,a1=.9,a2=0.)},out)

    def test_rejects_changed_unknown_support_or_invalid_distribution(self):
        for target in [dict(a0=.45,a1=.45,a2=.1),dict(a0=.5,a1=.6,a2=0.),dict(a0=float('nan'),a1=.5,a2=0.)]:
            with tempfile.TemporaryDirectory() as temp:
                parent,out,_,mc,_=self.fixture(Path(temp))
                with self.assertRaisesRegex(ValueError,'support|normalization'):
                    materialize(parent,{position(mc,'mc'):target},out)


if __name__=='__main__':unittest.main()
