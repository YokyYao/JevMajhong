import importlib.util
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

spec=importlib.util.spec_from_file_location('data_catalog',Path(__file__).resolve().parents[2]/'tools/data_catalog.py')
catalog=importlib.util.module_from_spec(spec);spec.loader.exec_module(catalog)


def example(group,view='native',label='skip'):
    return dict(state=dict(seat=0,melds=[[],[],[],[]],hand=['2m','3m']),
        questions={'action':dict(criteria=dict(skip={'type':'none'},call={'type':'chi','consumed':['2m','3m']}),label=label)},
        _meta=dict(group_id=group,id=group+'/0:'+view,parent_id=group+'/0',view=view,split='train'))


class DataCatalogTests(unittest.TestCase):
    def test_inventory_deduplicates_views_counts_passes_and_never_opens_test_payload(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);directory=root/'data/first';directory.mkdir(parents=True)
            rows=[example('training'),example('training','screen'),example('held_elsewhere')]
            train=directory/'train.jsonl';train.write_text(''.join(json.dumps(r)+'\n' for r in rows))
            (directory/'test.jsonl').write_bytes(b'INVALID SEALED PAYLOAD')
            (directory/'provenance.json').write_text(json.dumps(dict(complete=True,
                files={'train':catalog.sha(train)},matches=dict(train=['training','held_elsewhere'],test=['sealed']))))
            other=root/'data/incomplete';other.mkdir()
            (other/'train.jsonl').write_text('INVALID UNFINISHED PAYLOAD')
            (other/'provenance.json').write_text(json.dumps(dict(complete=False,matches={'validation':['held_elsewhere']})))
            result=catalog.run(dict(roots=[str(root)],out=str(root/'out')))
            self.assertEqual(result['all_indexed']['rows'],3)
            self.assertEqual(result['all_indexed']['known_positions'],2)
            self.assertEqual(result['training_candidates_before_quality_selection']['rows'],2)
            self.assertFalse(result['test_payloads_opened'])
            self.assertEqual(result['cross_split_groups_quarantined'],1)
            db=sqlite3.connect(root/'out/index-private.sqlite')
            self.assertEqual(db.execute('SELECT DISTINCT observed_kind FROM entries').fetchall(),[('none',)])
            self.assertEqual(db.execute('SELECT DISTINCT phase FROM entries').fetchall(),[('call_window',)])
            self.assertEqual(db.execute('SELECT count(DISTINCT request_hash) FROM entries').fetchone()[0],1)
            db.close()

    def test_unrepaired_call_menu_is_flagged_instead_of_silently_relabelled(self):
        request=example('g');request['questions']['action']['criteria']['call']['consumed']=[]
        phase,kinds,valid,reason=catalog.classify(request)
        self.assertEqual(phase,'call_window');self.assertIn('none',kinds)
        self.assertFalse(valid);self.assertEqual(reason,'consumed_count_requires_review')
        self.assertEqual(request['questions']['action']['criteria']['call']['consumed'],[])

    def test_future_branch_is_not_misclassified_as_expert_gold(self):
        db=sqlite3.connect(':memory:');catalog.initialize(db)
        row=dict(episode_key=[100,0,5,1],action=1)
        catalog.add(db,0,dict(method='paired_future_ev_v1'),'train',example('ignored'),{'file':'private'},native_row=row)
        self.assertEqual(db.execute('SELECT label_type,observed_kind,group_key FROM entries').fetchone(),
            ('paired_future_branch_not_gold','chi','native:100'))
        db.close()

    def test_kakan_uses_existing_pon_in_serialized_menu_and_added_tile_in_pai(self):
        r=example('g');r['questions']['action']['criteria']['call']=dict(type='kakan',pai='F',consumed=['F']*3)
        phase,kinds,valid,reason=catalog.classify(r)
        self.assertEqual(phase,'kan_window');self.assertTrue(valid);self.assertEqual(reason,'')
        r['questions']['action']['criteria']['call']['consumed']=['F']
        self.assertFalse(catalog.classify(r)[2])

    def test_reclassification_copies_completed_index_and_preserves_holdout_block(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);d=root/'data/source';d.mkdir(parents=True)
            row=example('g');row['questions']['action']['criteria']['call']=dict(type='kakan',pai='F',consumed=['F']*3)
            f=d/'train.jsonl';f.write_text(json.dumps(row)+'\n')
            (d/'provenance.json').write_text(json.dumps(dict(complete=True,matches={'train':['g'],'test':['g']},files={'train':catalog.sha(f)})))
            original=root/'first';catalog.run(dict(roots=[str(root)],out=str(original)))
            # Reproduce the old classifier's flag, as if it came from v1.
            db=sqlite3.connect(original/'index-private.sqlite');db.execute("UPDATE entries SET schema_valid=0,reason='consumed_count_requires_review'");db.commit();db.close()
            result=catalog.read(original/'result.json');result['private_index_sha256']=catalog.sha(original/'index-private.sqlite');catalog.write(original/'result.json',result)
            before=catalog.sha(original/'index-private.sqlite')
            revised=catalog.reclassify(dict(input_catalog=str(original),out=str(root/'revised')))
            self.assertEqual(revised['schema_records_corrected'],1)
            self.assertEqual(revised['training_candidates_before_quality_selection']['rows'],0)
            self.assertEqual(catalog.sha(original/'index-private.sqlite'),before)
            self.assertFalse(revised['source_payloads_changed'])


if __name__=='__main__':unittest.main()
