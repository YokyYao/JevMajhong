"""Attach parent targets after pinned augmentation/encoding via the epoch map."""
import copy
import math
from tools import curriculum_ri_parent_anchor_v1 as anchor
from tools import curriculum_progression_parent_loss_v1 as retained
from tools import curriculum_ri_inverse_repeat_loss_v1 as inverse
from tools.curriculum_ri_replay_epoch_adapter_v1 import EpochReplay,POSITION
from tools.audit_curriculum_basic_teacher import fingerprint

METHOD='parent_RI_targets_after_actual_encoding_from_frozen_epoch_originals_v2'


def attach(encoded,original,targets):
    identifier=original['_meta']['id']
    for q in encoded['questions']:
        raw=original['questions'][q['qid']]
        if (q.get(retained.MARKER)!=retained.METHOD or q.get(inverse.MARKER)!=inverse.METHOD
                or q.get(inverse.IDENTIFIER)!=identifier or q['src']!=raw['src']
                or set(q['keys'])!=set(raw['criteria']) or q['keys'][q['label']]!=raw['label']
                or anchor.FIELD in q or anchor.MARKER in q):
            raise ValueError('Pristine encoded weighted question and exact resolved original required')
        if q['src'] in anchor.RI_SOURCES:
            target=targets.get(identifier)
            if (target is None or target['source']!=q['src'] or target['label']!=raw['label']
                    or target['keys']!=frozenset(q['keys']) or target['visible_sha256']!=fingerprint(original)
                    or not math.isfinite(target['probability']) or not 0<target['probability']<1):
                raise ValueError('Complete exact TRAIN parent target required')
            q[anchor.FIELD]=dict(probability=target['probability'],reach=q['keys'].index(target['reach_key']))
        else:q[anchor.FIELD]=None
        q[anchor.MARKER]=anchor.METHOD
    return encoded


class AnchoredEpochReplay(EpochReplay):
    def __init__(self,rows,plans,targets):
        super().__init__(rows,plans)
        expected={r['_meta']['id'] for r in rows if r['questions']['action']['src'] in anchor.RI_SOURCES}
        if set(targets)!=expected:raise ValueError('Every and only actual RI TRAIN original needs a target')
        self.targets=copy.deepcopy(targets)

    def encode_batch(self,original,model,tok,args,chunk,epoch):
        positions=[r['_meta'].get(POSITION) for r in chunk]
        # The existing adapter validates contiguous positions, zero augmentation
        # settings, exact labels and weights before returning encoded variants.
        variants=super().encode_batch(original,model,tok,args,chunk,epoch)
        for position,variant in zip(positions,variants):
            index=self.plans[epoch]['indices'][position]
            attach(variant.rec,self.rows[index],self.targets)
        return variants
