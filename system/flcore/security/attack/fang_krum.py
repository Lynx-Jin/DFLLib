"""Oracle AGR-tailored Fang-style attack against Krum.

The attack intentionally observes current uploaded benign updates to test
whether Krum selects a colluding malicious update.  It is suitable for
evaluating a known Krum defense, not for claiming the AGR-agnostic threat model.
"""

from typing import Iterable, Sequence

import torch

from flcore.security.attack.minmax_minsum import AttackStats, MinMaxMinSumAttack


class FangKrumAttack:
    """Search the largest inverse-sign perturbation selected by Krum."""

    def __init__(self, assumed_malicious_ratio=0.2, lambda_init=10.0,
                 tolerance=1e-5, max_steps=80):
        if not 0 <= assumed_malicious_ratio < 0.5:
            raise ValueError("Fang-Krum assumed malicious ratio must be in [0, 0.5)")
        if lambda_init <= 0 or tolerance <= 0 or max_steps <= 0:
            raise ValueError("Fang-Krum search parameters must be positive")
        self.assumed_malicious_ratio = float(assumed_malicious_ratio)
        self.lambda_init = float(lambda_init)
        self.tolerance = float(tolerance)
        self.max_steps = int(max_steps)

    @staticmethod
    def _krum_scores(updates, assumed_malicious):
        count = updates.shape[0]
        neighbors = count - assumed_malicious - 2
        if count < 2 * assumed_malicious + 3 or neighbors <= 0:
            raise ValueError("Fang-Krum requires n >= 2f + 3 for its configured defense budget")
        distances = torch.cdist(updates, updates, p=2).square()
        distances.fill_diagonal_(float("inf"))
        return torch.topk(distances, k=neighbors, largest=False, dim=1).values.sum(dim=1)

    def apply(self, global_model, uploaded_ids: Sequence[int], uploaded_models: Sequence,
              malicious_ids: Iterable[int]):
        malicious_ids = set(malicious_ids)
        malicious_positions = [index for index, client_id in enumerate(uploaded_ids)
                               if client_id in malicious_ids]
        benign_positions = [index for index, client_id in enumerate(uploaded_ids)
                            if client_id not in malicious_ids]
        if not malicious_positions or not benign_positions:
            return None
        clean_updates = torch.stack([
            MinMaxMinSumAttack._flatten_model_update(global_model, model)
            for model in uploaded_models
        ]).to(dtype=torch.float32)
        benign_mean = clean_updates[benign_positions].mean(dim=0)
        direction = torch.sign(benign_mean)
        assumed = int(round(len(uploaded_models) * self.assumed_malicious_ratio))

        def candidate_for(value):
            return benign_mean - value * direction

        def feasible(value):
            trial = clean_updates.clone()
            trial[malicious_positions] = candidate_for(value)
            winner = int(torch.argmin(self._krum_scores(trial, assumed)).item())
            return winner in malicious_positions

        low, high = 0.0, self.lambda_init
        if not feasible(low):
            # Even the benign reference is not selected by Krum this round;
            # do not overwrite updates with a candidate that misses the target.
            return None
        while feasible(high) and high < self.lambda_init * (2 ** (self.max_steps // 2)):
            low, high = high, high * 2.0
        for _ in range(self.max_steps):
            if high - low <= self.tolerance:
                break
            middle = (low + high) / 2.0
            if feasible(middle):
                low = middle
            else:
                high = middle
        malicious_update = candidate_for(low)
        trial = clean_updates.clone()
        trial[malicious_positions] = malicious_update
        scores = self._krum_scores(trial, assumed)
        winner_score = scores[torch.argmin(scores)]
        for position in malicious_positions:
            MinMaxMinSumAttack._write_model_update(
                global_model, uploaded_models[position], malicious_update)
        return AttackStats("fang_krum_oracle", len(malicious_positions), low, 0.0,
                           float(winner_score.detach().cpu()))
