"""Stage2-native call utilities fitted from actual paired East terminal points."""
import copy
from threading import RLock

import torch
from call_policy import is_call_request
from diagonal_value import pointer_products
from guide_call_features import NAMES, VERSION, strategy_features
from guide_call_value import utility_preferences as tie_break_preferences
from stage2_ablation import FullStage2Client


def point_features(head, decide, options, request):
    # Match the collector/fitter's float32 public features before promoting the
    # complete feature matrix to float64 for the small point-value regression.
    latent = pointer_products(head, decide.float(), options.float())
    public = torch.tensor(strategy_features(request), dtype=torch.float32, device=latent.device)
    if latent.shape[0] != public.shape[0]: raise ValueError('Point feature action alignment differs')
    return torch.cat((latent, public), dim=1).double()


class FeaturePointClient:
    """Change chi/pon/pass only; preserve oldB ordinary and rare decisions."""
    def __init__(self, ordinary, candidate, checkpoint):
        if (checkpoint['strategy_feature_version'] != VERSION
                or checkpoint['strategy_feature_names'] != list(NAMES)
                or set(checkpoint['scope']) != {'chi', 'pon', 'none'}
                or checkpoint['target_scale'] != 1000.):
            raise ValueError('Unexpected point-value feature schema, scope or units')
        self.ordinary = ordinary
        self.teacher = FullStage2Client(candidate)
        self.head = candidate.runtime.model.head
        self.head.eval().requires_grad_(False)
        self.coefficient = checkpoint['coefficient'].to(device=next(self.head.parameters()).device, dtype=torch.float64)
        if (self.coefficient.shape != (self.head.q.out_features + len(NAMES),)
                or not torch.isfinite(self.coefficient).all()):
            raise ValueError('Invalid point-value coefficient')
        self.lock = RLock(); self.active = False; self.inputs = []; self.checks = 0; self.last = None
        self.hook = self.head.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        if self.active: self.inputs.append(tuple(t.detach().clone().float() for t in inputs))

    @torch.no_grad()
    def system_one(self, **request):
        if not is_call_request(request): return self.ordinary.system_one(**request)
        if request['state'].get('screen_observation'):
            raise ValueError('Point-value screenshot inference is not validated')
        from jevany.api import validate_distribution
        with self.lock:
            self.inputs.clear(); self.last = None; self.active = True
            try: result = self.teacher.system_one(**request)
            finally: self.active = False
            if len(self.inputs) != 1: raise ValueError('Missing unique stage2-native head capture')
            decide, options = self.inputs.pop()
            criteria = request['questions']['action']['criteria']; keys = list(criteria)
            native, _ = validate_distribution(result['answers']['action']['probabilities'], keys)
            replayed = self.head(decide, options).softmax(-1)
            if (len(replayed) != len(keys) or not torch.allclose(
                    torch.tensor(native, device=replayed.device), replayed, atol=2e-5, rtol=2e-4)):
                raise ValueError('Stage2 pointer features do not reproduce native probabilities')
            features = point_features(self.head, decide, options, request)
            teacher_index = int(replayed.argmax())
            # The fitted target is action minus frozen-teacher points. Form
            # the same float32 difference as the immutable fitter, then use
            # float64 coefficients. Teacher utility is exactly zero.
            relative = (features.float() - features.float()[teacher_index]).double()
            values = relative @ self.coefficient
            pass_index = next(i for i, c in enumerate(criteria.values()) if c['type'] == 'none')
            # The shared helper accepts a preferred tie index. Here a neutral
            # tie keeps the frozen teacher, rather than imposing a pass bonus.
            preferences, chosen = tie_break_preferences(values, teacher_index)
            output = copy.deepcopy(result)
            output['answers']['action'] = {'probabilities': dict(zip(keys, preferences))}
            self.last = {'decide': decide.cpu(), 'options': options.cpu(), 'features': features.cpu(),
                'relative_features': relative.cpu(), 'values': values.cpu(), 'chosen': chosen,
                'pass_index': pass_index, 'teacher_index': teacher_index,
                'teacher_probabilities': replayed.cpu()}
            self.checks += 1
            return output

    def close(self): self.hook.remove()
