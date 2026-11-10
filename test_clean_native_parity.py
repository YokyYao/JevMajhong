import json
from pathlib import Path
import tempfile
import unittest

from scripts.evaluate_clean_policy import native_parity


class Client:
    def system_one(self,**request):
        return {'answers':{'action':{'probabilities':{'a0':.7,'a1':.3}}}}


class NativeParityTests(unittest.TestCase):
    def test_native_only_curriculum_still_requires_all_three_action_phases(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);data=root/'data';data.mkdir();run=root/'run'
            evaluation=run/'training-run/training_eval/step-000050/development';evaluation.mkdir(parents=True)
            records=[];rows=[]
            for phase in ('closed_discard','open_discard','reaction'):
                for i in range(2):
                    identity=f'{phase}/{i}'
                    records.append({'state':{},'questions':{'action':{'type':'choice','instructions':'test',
                        'criteria':{'a0':{'type':'dahai'},'a1':{'type':'dahai'}},'label':'a0'}},
                        '_meta':{'id':identity,'phase':phase,'view':'native'}})
                    rows.append({'id':identity,'keys':['a0','a1'],'p':[.7,.3]})
            (data/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            (evaluation/'rows.json').write_text(json.dumps(rows))
            preparation={'config':{'data':str(data)}}
            result=native_parity(Client(),run,preparation,50,2,.0005,['native'])
            self.assertEqual(result['checked'],6)
            self.assertEqual(result['max_absolute_probability_error'],0)
            (data/'validation.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records if r['_meta']['phase']!='reaction'))
            with self.assertRaises(ValueError):native_parity(Client(),run,preparation,50,2,.0005,['native'])


if __name__=='__main__':unittest.main()
