"""Training-only choice likelihood with fixed known inclusion offsets."""
import math
from call_sampling_prior import option_class


class SamplingLossHooks:
    """Attach public-option offsets after encoding; keep model/serving unchanged.

For inclusion probability s(x,y), the sampled conditional distribution is
softmax(z + log s) when raw z represents the source-population logits.
This is a modeling assumption, not proof of score or calibration improvement.
"""
    def __init__(self, trainer, inverse_weights):
        self.trainer = trainer
        self.weights = inverse_weights
        self.reaction_records = self.ordinary_records = self.adjusted_losses = 0

    def annotate(self, chunk, variants):
        if len(chunk) != len(variants):
            raise ValueError('One unaugmented variant per record required')
        for request, variant in zip(chunk, variants):
            if len(variant.rec['questions']) != 1:
                raise ValueError('Single action question required')
            q = variant.rec['questions'][0]
            source = request['questions']['action']
            if q['qid'] != 'action' or set(q['keys']) != set(source['criteria']) or q['keys'][q['label']] != source['label']:
                raise ValueError('Encoded action alignment differs')
            if request['_meta']['phase'] != 'reaction':
                self.ordinary_records += 1
                continue
            offsets = [-math.log(self.weights[option_class(source['criteria'][key])]) for key in q['keys']]
            q['_sampling_log_inclusion'] = offsets
            self.reaction_records += 1
        return variants

    def __enter__(self):
        import torch
        self.original_encode = self.trainer.encode_batch
        self.original_loss = self.trainer.question_loss

        def encode(model, tok, args, chunk, epoch):
            if args.p_none or args.p_none_distract or args.p_distract or args.p_none_pair:
                raise ValueError('Sampling likelihood requires unchanged options/labels')
            return self.annotate(chunk, self.original_encode(model, tok, args, chunk, epoch))

        def loss(z, q, dev):
            offsets = q.get('_sampling_log_inclusion')
            if offsets is None:
                return self.original_loss(z, q, dev)
            if len(offsets) != len(z) or q.get('target') is not None or any(not math.isfinite(x) for x in offsets):
                raise ValueError('Known hard-label choice offsets required')
            self.adjusted_losses += 1
            adjusted = z + torch.tensor(offsets, device=z.device, dtype=z.dtype)
            return self.original_loss(adjusted, q, dev)

        self.trainer.encode_batch, self.trainer.question_loss = encode, loss
        return self

    def __exit__(self, *exc):
        self.trainer.encode_batch, self.trainer.question_loss = self.original_encode, self.original_loss
