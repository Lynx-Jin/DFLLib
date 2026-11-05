"""PyTorch port of xyq7/PoisonedFL, commit 266488e2cbe5953aab61712f315518546f457e55.

Port of byzantine.py:poisonedfl and the state bookkeeping in test_agr.py.
Only the basic attack is implemented, not supplementary PoisonedFL-Adapt.
The attacker sees global models and its own last attempted update only.
Project feedback default: 10 rounds. Use window=50 for the author setting.
FedSGD/Adam uses an explicit adam_proxy adaptation; it is not an exact
conversion of author model deltas through the nonlinear Adam optimizer.
"""

import math

import torch
from scipy.stats import binom


DEFAULT_POISONEDFL_WINDOW = 10


def flatten_model(model):
    return torch.cat([p.detach().reshape(-1) for p in model.parameters()])


def write_model(model, vector):
    offset = 0
    with torch.no_grad():
        for parameter in model.parameters():
            size = parameter.numel()
            parameter.copy_(vector[offset:offset + size].view_as(parameter))
            offset += size
    if offset != vector.numel():
        raise ValueError('Vector length does not match model')


class PoisonedFLAttack:
    def __init__(self, scale=8., window=DEFAULT_POISONEDFL_WINDOW, decay=.7, min_scale=.5, seed=0):
        if (not all(math.isfinite(x) for x in (scale, decay, min_scale))
                or scale <= 0 or min_scale <= 0 or not 0 < decay < 1
                or window < 1 or seed < 0):
            raise ValueError('Invalid PoisonedFL parameters')
        self.scale = float(scale)
        self.window = int(window)
        self.decay = float(decay)
        self.min_scale = float(min_scale)
        self.seed = int(seed)
        self.round = 0
        self.previous_global = None
        self.last_malicious_update = None
        self.window_anchor = None
        self.fixed_sign = None
        self.k_99 = None

    def propose(self, global_vector, active=True):
        """Observe W_t, then craft m_t. Bootstrap m_0=0, as in author code.

        For L>1, anchors are W_1 at t=L and W_(L+1) at t=2L,
        preserving the author's post-aggregation checkpoint bookkeeping.
        L=1 uses W_(t-1), so the feedback measures the last actual update.
        """
        current = global_vector.detach().to(dtype=torch.float32).clone()
        if current.ndim != 1 or not torch.isfinite(current).all():
            raise ValueError('Expected a finite flat global model')
        if self.fixed_sign is None:
            generator = torch.Generator(device='cpu').manual_seed(self.seed)
            self.fixed_sign = (2 * torch.randint(0, 2, current.shape, generator=generator)
                               - 1).to(current)
            self.k_99 = int(binom.ppf(.99, current.numel(), .5))
        if current.shape != self.fixed_sign.shape:
            raise ValueError('Global model dimension changed')
        stats = dict(round=self.round, phase='bootstrap' if self.round == 0 else 'attack',
                     scale_before=self.scale, scale=self.scale, scale_adjusted=False,
                     active=bool(active), dimensions=current.numel(), k_99=self.k_99,
                     aligned_dimensions=None, history_norm=0., update_norm=0.,
                     window=self.window, feedback_checked=False)
        if self.previous_global is None:
            update = torch.zeros_like(current)
        else:
            history = current - self.previous_global
            history_norm = torch.linalg.vector_norm(history)
            if (self.round - 1) % self.window == 0:
                # At L=1 the checkpoint and feedback conditions coincide.
                # Anchoring at current would make every difference zero and
                # test sign(W_t) via the author's zero-coordinate fallback.
                self.window_anchor = (self.previous_global if self.window == 1 else current).clone()
            if active and self.round % self.window == 0:
                total_update = current - self.window_anchor
                # Preserve author's special treatment of zero coordinates.
                total_update = torch.where(total_update == 0, current, total_update)
                aligned = int((total_update.sign() == self.fixed_sign).sum().item())
                if aligned < self.k_99 and self.scale * self.decay >= self.min_scale:
                    self.scale *= self.decay
                    stats['scale_adjusted'] = True
                stats.update(feedback_checked=True, aligned_dimensions=aligned)
            last = self.last_malicious_update
            last_norm = torch.linalg.vector_norm(last)
            coordinate_scale = (history - last * history_norm / (last_norm + 1e-9)).abs()
            deviation = coordinate_scale * self.fixed_sign / (
                torch.linalg.vector_norm(coordinate_scale) + 1e-9)
            update = self.scale * history_norm * deviation
            stats['history_norm'] = float(history_norm.item())
        if active:
            self.last_malicious_update = update.clone()
        elif self.last_malicious_update is None:
            self.last_malicious_update = torch.zeros_like(current)
        self.previous_global = current
        self.round += 1
        stats.update(scale=self.scale, update_norm=float(torch.linalg.vector_norm(update).item()))
        return update, stats

    def apply(self, global_model, client_ids, client_models, malicious_ids,
              gradient_learning_rate=None, gradient_mode='sgd_delta'):
        if gradient_mode not in ('sgd_delta', 'adam_proxy'):
            raise ValueError('Unknown PoisonedFL gradient mapping')
        if gradient_mode == 'adam_proxy' and gradient_learning_rate is None:
            raise ValueError('Adam gradient proxy requires a FedSGD carrier')
        if gradient_learning_rate is not None and (
                not math.isfinite(gradient_learning_rate) or gradient_learning_rate <= 0):
            raise ValueError('Gradient learning rate must be finite and positive')
        controlled = set(malicious_ids)
        active_ids = [int(cid) for cid in client_ids if cid in controlled]
        current = flatten_model(global_model)
        update, stats = self.propose(current, active=bool(active_ids))
        # Compute one payload once: no independent per-client perturbation.
        # FedSGD stores W+g, then applies W <- W-lr*mean(g). The attack
        # proposes a MODEL DELTA m, so g_m=-m/lr, rather than g_m=m.
        if gradient_learning_rate is None:
            uploaded_update = update
        elif gradient_mode == 'adam_proxy':
            # Explicit FedSGD/Adam adaptation: the history-scaled descent
            # direction is a gradient-magnitude PROXY q, and g_m=-q. Adam
            # applies its own moment normalization. Do not divide by lr or
            # claim that Adam's resulting parameter delta equals q.
            uploaded_update = -update
        else:
            uploaded_update = -update / gradient_learning_rate
        submitted_model = current + uploaded_update
        for cid, model in zip(client_ids, client_models):
            if cid in controlled:
                write_model(model, submitted_model)
        stats.update(active_malicious_ids=active_ids, identical_updates=True,
                     uploaded_vector=('local_model' if gradient_learning_rate is None
                                      else 'global_plus_gradient_carrier'),
                     gradient_learning_rate=gradient_learning_rate,
                     gradient_mapping=('none' if gradient_learning_rate is None else gradient_mode),
                     model_delta_equivalent=(gradient_learning_rate is None or gradient_mode == 'sgd_delta'),
                     uploaded_update_norm=float(torch.linalg.vector_norm(uploaded_update).item()))
        return stats
