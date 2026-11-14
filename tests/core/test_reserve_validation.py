import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from jevmahjong.core.io import digest
from jevmahjong.data.reserve_validation import reserve


def fixture(root, overlap=False):
    parent=root/'pool';parent.mkdir();source=root/'expert';source.mkdir();family=root/'family';family.mkdir()
    (source/'train.jsonl').write_text('frozen audited source\n')
    (source/'provenance.json').write_text(json.dumps(dict(matches={'train':['reserved','seen']})))
    train=family/'train.jsonl';train.write_text(json.dumps({'_meta':{'group_id':'replay:seen'}})+'\n')
    val=family/'validation.jsonl';val.write_text(json.dumps({'_meta':{'group_id':'replay:oldval'}})+'\n')
    protocol=family/'protocol.json';protocol.write_text(json.dumps(dict(trainer_args=dict(data=str(train),eval_data=str(val)),
        input_sha256={str(train):digest(train),str(val):digest(val)})))
    (family/'initialization-audit.json').write_text(json.dumps(dict(old_adapters_loaded=False,external_residual_networks=0)))
    db=sqlite3.connect(parent/'pool-private.sqlite')
    db.execute('CREATE TABLE cases(position TEXT,split TEXT,group_key TEXT,phase TEXT,label_type TEXT,choice TEXT,view TEXT,priority INTEGER,payload TEXT,PRIMARY KEY(position,split))')
    for pos,split,group,phase,choice,priority in [
        ('s','train','seen','closed_discard','dahai',50),
        ('k','train','reserved','kan_window','ankan',80),
        ('c','train','reserved','call_window','none',50),
        ('v','validation','oldval','call_window','pon',50)]:
        db.execute('INSERT INTO cases VALUES(?,?,?,?,?,?,?,?,?)',
            (pos,split,'replay:'+group,phase,'expert_replay',choice,'native',priority,json.dumps(dict(id=pos,label=choice))))
    db.commit();db.close()
    (parent/'result.json').write_text(json.dumps(dict(status='raw_request_pool_ready_not_yet_balanced',
        requires_masked_soft_target_loss=True,pool_sha256=digest(parent/'pool-private.sqlite'),
        config=dict(sources=[dict(path=str(source),priority=80),dict(path=str(root/'other'),priority=50)]))))
    group='replay:seen' if overlap else 'replay:reserved'
    manifest=root/'reservation.json';manifest.write_text(json.dumps(dict(schema='family-unseen-expert-prefix-validation-v1',
        groups=[group],family_runs={str(family):digest(protocol)},
        family_input_sha256={str(protocol):digest(protocol),str(train):digest(train),str(val):digest(val)},
        source_directory=str(source),source_train_sha256=digest(source/'train.jsonl'),
        source_provenance_sha256=digest(source/'provenance.json'))))
    marker=root/'holdouts.json';marker.write_text(json.dumps(dict(complete=False,
        matches=dict(train=[],validation=[group.removeprefix('replay:')],test=[]),reservation_sha256=digest(manifest))))
    return dict(parent_pool=str(parent),parent_pool_sha256=digest(parent/'pool-private.sqlite'),
        parent_result_sha256=digest(parent/'result.json'),reservation=str(manifest),reservation_sha256=digest(manifest),
        holdout_metadata=str(marker),holdout_metadata_sha256=digest(marker),out=str(root/'derived'))


class ReserveValidationTests(unittest.TestCase):
    def test_moves_expert_windows_excludes_other_same_match_training_and_keeps_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp));before=digest(Path(cfg['parent_pool'])/'pool-private.sqlite');result=reserve(cfg)
            proof=result['validation_reservation']
            self.assertEqual(proof['after'],dict(train=1,validation=2))
            self.assertEqual(proof['expert_kan_records_moved'],1)
            self.assertEqual(proof['other_source_train_records_excluded'],1)
            self.assertFalse(proof['payloads_and_labels_changed']);self.assertEqual(result['groups_overlap'],0)
            self.assertEqual(digest(Path(cfg['parent_pool'])/'pool-private.sqlite'),before)
            db=sqlite3.connect(Path(cfg['out'])/'pool-private.sqlite')
            self.assertEqual(db.execute('SELECT split,choice,payload FROM cases WHERE position=?',('k',)).fetchone(),
                ('validation','ankan',json.dumps(dict(id='k',label='ankan'))))
            self.assertEqual(db.execute("SELECT COUNT(*) FROM cases WHERE split='train' AND group_key='replay:reserved'").fetchone()[0],0)
            db.close()

    def test_declared_held_group_cannot_have_appeared_in_actual_family_data(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp),overlap=True)
            with self.assertRaisesRegex(ValueError,'Reserved group was seen by model family'):
                reserve(cfg)

    def test_parent_data_change_is_rejected_before_partition(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg=fixture(Path(tmp));db=sqlite3.connect(Path(cfg['parent_pool'])/'pool-private.sqlite')
            db.execute("UPDATE cases SET payload='changed'");db.commit();db.close()
            with self.assertRaisesRegex(ValueError,'Bound validation reservation input changed'):
                reserve(cfg)


if __name__=='__main__':unittest.main()
