"""Full25-source epoch mapping; attach fixed RI targets after real encoding."""
import copy
import math
from types import FunctionType

from tools import curriculum_ri_parent_anchor_v1 as anchor
from tools import curriculum_yaku_parent_loss_v1 as yaku
from tools import curriculum_ri_inverse_repeat_loss_v1 as inverse
from tools import curriculum_yaku_ri_anchor_loss_v1 as objective
from tools.curriculum_ri_replay_epoch_adapter_v1 import EpochReplay, POSITION
from tools.audit_curriculum_basic_teacher import fingerprint

METHOD = 'yaku25_inverse_repeat_epoch_and_exact_RI_targets_after_encoding_v1'


def attach_targets(encoded, original, targets):
    identifier = original['_meta']['id']
    for question in encoded['questions']:
        raw = original['questions'][question['qid']]
        if (question.get(yaku.MARKER) != yaku.METHOD
                or question.get(inverse.MARKER) != objective.WEIGHT_METHOD
                or question.get(inverse.IDENTIFIER) != identifier or question['src'] != raw['src']
                or set(question['keys']) != set(raw['criteria'])
                or question['keys'][question['label']] != raw['label']
                or anchor.FIELD in question or anchor.MARKER in question):
            raise ValueError('Pristine encoded weighted yaku question and resolved original required')
        if question['src'] in anchor.RI_SOURCES:
            target = targets.get(identifier)
            if (target is None or target['source'] != question['src'] or target['label'] != raw['label']
                    or target['keys'] != frozenset(question['keys'])
                    or target['visible_sha256'] != fingerprint(original)
                    or not math.isfinite(target['probability']) or not 0 < target['probability'] < 1):
                raise ValueError('Complete exact TRAIN RI identity/input/menu/target required')
            question[anchor.FIELD] = dict(probability=target['probability'],
                reach=question['keys'].index(target['reach_key']))
        else:
            question[anchor.FIELD] = None
        question[anchor.MARKER] = anchor.METHOD
    return encoded


class YakuAnchoredEpochReplay(EpochReplay):
    def __init__(self, rows, plans, targets):
        constructor = FunctionType(EpochReplay.__init__.__code__,
            dict(EpochReplay.__init__.__globals__, MASS=dict(yaku.MASS)),
            EpochReplay.__init__.__name__, EpochReplay.__init__.__defaults__,
            EpochReplay.__init__.__closure__)
        constructor(self, rows, plans)
        expected = {row['_meta']['id'] for row in rows
            if row['questions']['action']['src'] in anchor.RI_SOURCES}
        if set(targets) != expected:
            raise ValueError('Every and only actual basic-RI TRAIN row needs a frozen target')
        self.targets = copy.deepcopy(targets)

    def encode_batch(self, original, model, tok, args, chunk, epoch):
        positions = [row['_meta'].get(POSITION) for row in chunk]
        # Preserve the numerical exposure/label/coverage code object, changing
        # only its post-encoder weight-admission hook for the25-source schema.
        encode = FunctionType(EpochReplay.encode_batch.__code__,
            dict(EpochReplay.encode_batch.__globals__, attach=objective.attach_weight),
            EpochReplay.encode_batch.__name__, EpochReplay.encode_batch.__defaults__,
            EpochReplay.encode_batch.__closure__)
        variants = encode(self, original, model, tok, args, chunk, epoch)
        for position, variant in zip(positions, variants):
            index = self.plans[epoch]['indices'][position]
            attach_targets(variant.rec, self.rows[index], self.targets)
        return variants

    def finish(self):
        result = super().finish()
        result['mapping_body_method'] = result['method']
        result['method'] = METHOD
        return result
