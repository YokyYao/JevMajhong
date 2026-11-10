"""Native Jev call interface for the paired-point predictor; evaluation only."""
import copy
import math
from threading import RLock

import torch
from call_policy import CallRouter, is_call_request
from diagonal_value import pointer_products
from guide_call_features import NAMES, VERSION, strategy_features


def point_features(head, decide, options, request):
    latent = pointer_products(head, decide.float(), options.float()).double()
    public = torch.tensor(strategy_features(request), dtype=torch.float64, device=latent.device)
    if len(latent) != len(public): raise ValueError('Native action/strategy alignment mismatch')
    return torch.cat((latent, public), dim=1)


def utility_preferences(values, pass_index):
    """Softmax action preferences; an exact neutral tie prefers pass.

    The one-ULP tie adjustment is not a training reward or point bonus.
    Returned numbers are preferences, never calibrated win probabilities.
    """
    if values.ndim != 1 or not torch.isfinite(values).all(): raise ValueError('Nonfinite point utility')
    best = int(values.argmax())
    if values[pass_index] == values[best]: best = pass_index
    p = values.softmax(0).tolist()
    if any(i != best and v >= p[best] for i, v in enumerate(p)):
        p[best] = math.nextafter(max(p), math.inf)
    return p, best


class PointValueCallRouter:
    """Use the frozen Jev representation plus learned public-state covariates.

    All ordinary decisions use the original client. Only native chi/pon/none
    choices use learned terminal-point utilities. Screenshot parity is pending.
    """
    def __init__(self, client, teacher_head, checkpoint):
        if checkpoint['strategy_feature_version'] != VERSION or checkpoint['strategy_feature_names'] != list(NAMES):
            raise ValueError('Point-predictor feature schema mismatch')
        if set(checkpoint['scope']) != {'chi','pon','none'}: raise ValueError('Wrong action scope')
        device = next(teacher_head.parameters()).device
        self.coefficient = checkpoint['coefficient'].to(device=device, dtype=torch.float64)
        if self.coefficient.shape != (teacher_head.q.out_features + len(NAMES),) or not torch.isfinite(self.coefficient).all():
            raise ValueError('Point coefficient shape or values invalid')
        self.client, self.head = client, teacher_head
        self.call_router = CallRouter(client, teacher_head)
        self.lock = RLock(); self.active = False; self.inputs = []; self.checks = 0; self.last = None
        self.hook = teacher_head.register_forward_pre_hook(self._capture)

    def _capture(self, module, inputs):
        if self.active: self.inputs.append(tuple(t.detach().clone().float() for t in inputs))

    @torch.no_grad()
    def system_one(self, **request):
        if not is_call_request(request): return self.client.system_one(**request)
        if request['state'].get('screen_observation'):
            raise ValueError('Point-value screenshot inference has not been validated')
        from jevany.api import validate_distribution
        with self.lock:
            self.inputs.clear(); self.last = None; original = self.client.runtime.model.head
            self.active = True
            try: result = self.call_router.system_one(**request)
            finally: self.active = False
            if self.client.runtime.model.head is not original: raise ValueError('Discard head was not restored')
            if len(self.inputs) != 1: raise ValueError('Expected one native call pointer capture')
            decide, options = self.inputs.pop()
            criteria = request['questions']['action']['criteria']; keys = list(criteria)
            native, _ = validate_distribution(result['answers']['action']['probabilities'], keys)
            teacher = self.head(decide, options).softmax(-1)
            if len(teacher) != len(keys) or not torch.allclose(torch.tensor(native,device=teacher.device), teacher,atol=2e-5,rtol=2e-4):
                raise ValueError('Native teacher/feature probabilities differ')
            matrix = point_features(self.head, decide, options, request)
            values = matrix @ self.coefficient
            pass_index = next(i for i, c in enumerate(criteria.values()) if c['type'] == 'none')
            p, chosen = utility_preferences(values, pass_index)
            output = copy.deepcopy(result)
            output['answers']['action'] = {'probabilities': dict(zip(keys,p))}
            self.checks += 1
            self.last = {'decide':decide.cpu(),'options':options.cpu(),'features':matrix.cpu(),
                         'values':values.cpu(),'chosen':chosen,'pass_index':pass_index}
            return output

    def close(self): self.hook.remove()
