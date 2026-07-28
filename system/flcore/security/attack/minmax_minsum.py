"""Aggregation-agnostic Min-Max and Min-Sum model-update attacks.

The attacker does not observe honest-client updates. Instead, the clean local
updates produced by malicious clients in the current round serve as proxies for
the benign update distribution. One optimized update is assigned to every
active malicious client, matching the colluding construction in NDSS 2021.
"""

from collections import namedtuple
from typing import Iterable, Sequence

import torch


AttackStats = namedtuple(
    "AttackStats",
    ["attack", "malicious_clients", "lambda_value", "threshold", "score"],
)


class MinMaxMinSumAttack:
    """Create one shared malicious update from malicious proxy updates."""

    SUPPORTED_ATTACKS = {"minmax", "minsum"}
    SUPPORTED_DEVIATIONS = {"unit_vec", "sign", "std"}

    def __init__(self, attack: str, deviation: str = "std",
                 lambda_init: float = 10.0, tolerance: float = 1e-5,
                 max_steps: int = 80):
        attack = attack.lower().replace("-", "").replace("_", "")
        if attack not in self.SUPPORTED_ATTACKS:
            raise ValueError(f"Unsupported model-poisoning attack: {attack}")
        if deviation not in self.SUPPORTED_DEVIATIONS:
            raise ValueError(f"Unsupported perturbation direction: {deviation}")
        if lambda_init <= 0 or tolerance <= 0 or max_steps <= 0:
            raise ValueError("Search parameters must be positive")
        self.attack = attack
        self.deviation = deviation
        self.lambda_init = float(lambda_init)
        self.tolerance = float(tolerance)
        self.max_steps = int(max_steps)

    @staticmethod
    def _flatten_model_update(global_model, client_model):
        parts = [
            (client_param.detach() - global_param.detach()).reshape(-1)
            for global_param, client_param in zip(
                global_model.parameters(), client_model.parameters())
        ]
        if not parts:
            raise ValueError("Cannot attack a model with no trainable parameters")
        return torch.cat(parts)

    @staticmethod
    def _write_model_update(global_model, client_model, update):
        offset = 0
        with torch.no_grad():
            for global_param, client_param in zip(
                    global_model.parameters(), client_model.parameters()):
                size = global_param.numel()
                value = update[offset:offset + size].view_as(global_param)
                client_param.copy_(
                    global_param.detach() + value.to(
                        device=client_param.device, dtype=client_param.dtype))
                offset += size
        if offset != update.numel():
            raise ValueError("Flattened update does not match model parameters")

    def _deviation_vector(self, proxy_updates, proxy_mean):
        if self.deviation == "unit_vec":
            norm = torch.norm(proxy_mean)
            if norm <= torch.finfo(proxy_mean.dtype).eps:
                return torch.zeros_like(proxy_mean)
            return proxy_mean / norm
        if self.deviation == "sign":
            return torch.sign(proxy_mean)
        # correction=0 keeps a one-client proxy set finite.
        return torch.std(proxy_updates, dim=0, unbiased=False)

    def _reference_threshold(self, proxy_updates):
        distances = torch.cdist(proxy_updates, proxy_updates, p=2).square()
        if self.attack == "minmax":
            return distances.max()
        # The released implementation uses min. The paper's displayed Eq. (7)
        # prints max, but min makes the malicious update at least as central as
        # the most central proxy update and matches the public attack code.
        return distances.sum(dim=1).max()#改成max试一试

    def _score(self, candidate, proxy_updates):
        distances = torch.norm(
            proxy_updates - candidate.unsqueeze(0), dim=1).square()
        return distances.max() if self.attack == "minmax" else distances.sum()

    def _search_lambda(self, proxy_updates, proxy_mean, deviation, threshold):
        if torch.count_nonzero(deviation) == 0:
            score = self._score(proxy_mean, proxy_updates)
            return 0.0, proxy_mean, score

        def feasible(value):
            candidate = proxy_mean - value * deviation
            score = self._score(candidate, proxy_updates)
            return bool(score <= threshold)

        low, high = 0.0, self.lambda_init
        expansion_steps = 0
        while feasible(high) and expansion_steps < self.max_steps // 2:
            low, high = high, high * 2.0
            expansion_steps += 1

        if feasible(high):
            candidate = proxy_mean - high * deviation
            return high, candidate, self._score(candidate, proxy_updates)

        for _ in range(self.max_steps):
            if high - low <= self.tolerance:
                break
            mid = (low + high) / 2.0
            if feasible(mid):
                low = mid
            else:
                high = mid

        candidate = proxy_mean - low * deviation
        return low, candidate, self._score(candidate, proxy_updates)

    def apply(self, global_model, uploaded_ids: Sequence[int],
              uploaded_models: Sequence, malicious_ids: Iterable[int]):
        malicious_ids = set(malicious_ids)
        positions = [
            index for index, client_id in enumerate(uploaded_ids)
            if client_id in malicious_ids
        ]
        if not positions:
            return None

        # Only clean updates from active malicious clients are exposed here.
        proxy_updates = torch.stack([
            self._flatten_model_update(global_model, uploaded_models[index])
            for index in positions
        ]).to(dtype=torch.float32)
        proxy_mean = proxy_updates.mean(dim=0)
        deviation = self._deviation_vector(proxy_updates, proxy_mean)
        threshold = self._reference_threshold(proxy_updates)
        lambda_value, malicious_update, score = self._search_lambda(
            proxy_updates, proxy_mean, deviation, threshold)

        # All colluding clients submit exactly the same optimized update.
        for index in positions:
            self._write_model_update(
                global_model, uploaded_models[index], malicious_update)

        return AttackStats(
            attack=self.attack,
            malicious_clients=len(positions),
            lambda_value=lambda_value,
            threshold=float(threshold.detach().cpu()),
            score=float(score.detach().cpu()),
        )

