"""Cosine-aware Min-Max adaptations using only active malicious proxy updates.

Ported from the experiment implementation. cosine_oracle replaces the L2
feasibility oracle with a signed model/global cosine constraint; cosine keeps
the agnostic L2 cap. Neither reads benign uploads or clustering decisions.
"""
from collections import namedtuple
import numpy as np
import torch
from flcore.security.attack.minmax_minsum import MinMaxMinSumAttack
from flcore.security.attack.cosine_attack_search import strongest_cosine_point

AttackStats = namedtuple(
    "AttackStats",
    [
        "attack", "variant", "malicious_clients", "deviation",
        "minsum_threshold_mode", "lambda_value", "agnostic_lambda",
        "cosine_target", "base_cosine", "final_cosine",
        "cosine_target_reached", "threshold", "score",
        "perturbation_l2", "relative_perturbation", "search_cap_reached",
    ],
)


class CosineTailoredMinMaxAttack(MinMaxMinSumAttack):
    """One colluding upload on the fixed-direction Min-Max search ray."""

    def __init__(self, deviation='std', lambda_init=10., tolerance=1e-5,
                 max_steps=80, variant='cosine_oracle', cosine_target=None,
                 cosine_search_steps=256, cosine_quantile=.5):
        super().__init__('minmax', deviation, lambda_init, tolerance, max_steps)
        if variant not in ('cosine', 'cosine_oracle'):
            raise ValueError('Expected cosine or cosine_oracle variant')
        if cosine_target is not None and not -1 <= cosine_target <= 1:
            raise ValueError('cosine_target must be in [-1, 1]')
        if cosine_search_steps < 2 or not 0 <= cosine_quantile <= 1:
            raise ValueError('Invalid cosine search steps or proxy quantile')
        self.variant = variant
        self.cosine_target = cosine_target
        self.cosine_search_steps = int(cosine_search_steps)
        self.cosine_quantile = float(cosine_quantile)
        self.minsum_threshold_mode = 'min'  # Unused for Min-Max; audit compatibility.

    @staticmethod
    def flatten_model(model):
        parts = [
            parameter.detach().reshape(-1)
            for parameter in model.parameters()
        ]
        if not parts:
            raise ValueError("Model has no parameters")
        return torch.cat(parts)

    @staticmethod
    def cosine(first, second):
        denominator = torch.norm(first) * torch.norm(second)
        if denominator <= torch.finfo(first.dtype).eps:
            return 0.0
        return float(
            (torch.dot(first, second) / denominator).detach().cpu())

    def cosine_tailored_search(
            self, global_vector, mean, deviation, lambda_upper, target):
        """Keep the strongest feasible Min-Max point above a cosine target.

        The candidate remains on the standard Min-Max ray
        ``mean - lambda * deviation`` and lambda never exceeds the
        AGR-agnostic feasible value.  The cosine scan uses precomputed scalar
        dot products instead of repeatedly constructing full model vectors.
        """
        lambda_upper = max(0.0, float(lambda_upper))
        base_model = global_vector + mean
        direction = -deviation

        global_sq = float(torch.dot(global_vector, global_vector).detach().cpu())
        base_sq = float(torch.dot(base_model, base_model).detach().cpu())
        direction_sq = float(torch.dot(direction, direction).detach().cpu())
        base_global = float(
            torch.dot(base_model, global_vector).detach().cpu())
        direction_global = float(
            torch.dot(direction, global_vector).detach().cpu())
        base_direction = float(
            torch.dot(base_model, direction).detach().cpu())

        def cosine_at(value):
            numerator = base_global + value * direction_global
            model_sq = (
                base_sq
                + 2.0 * value * base_direction
                + value * value * direction_sq
            )
            denominator_sq = global_sq * max(model_sq, 0.0)
            if denominator_sq <= 1e-24:
                return 0.0
            result = numerator / (denominator_sq ** 0.5)
            return max(-1.0, min(1.0, result))

        mean_cosine = cosine_at(0.0)
        effective_target = mean_cosine if target is None else float(target)
        upper_cosine = cosine_at(lambda_upper)
        if upper_cosine >= effective_target:
            candidate = mean - lambda_upper * deviation
            return (
                lambda_upper, candidate, upper_cosine,
                effective_target, True,
            )

        # Do not assume that cosine is monotonic along an arbitrary Min-Max ray.
        feasible = []
        step = lambda_upper / float(self.cosine_search_steps)
        for index in range(self.cosine_search_steps + 1):
            value = step * index
            if cosine_at(value) >= effective_target:
                feasible.append(value)

        if not feasible:
            candidate = mean
            return 0.0, candidate, mean_cosine, effective_target, False

        best = max(feasible)
        high = min(lambda_upper, best + step)
        if high > best and cosine_at(high) < effective_target:
            low = best
            for _ in range(self.max_steps):
                if high - low <= self.tolerance:
                    break
                middle = (low + high) / 2.0
                if cosine_at(middle) >= effective_target:
                    low = middle
                else:
                    high = middle
            best = low

        candidate = mean - best * deviation
        return best, candidate, cosine_at(best), effective_target, True

    def apply(self, global_model, uploaded_ids, uploaded_models, malicious_ids):
        malicious_ids = set(malicious_ids)
        positions = [
            index for index, client_id in enumerate(uploaded_ids)
            if client_id in malicious_ids
        ]
        if not positions:
            return None

        clean_updates = torch.stack([
            self._flatten_model_update(global_model, uploaded_models[index])
            for index in positions
        ]).to(dtype=torch.float32)
        proxy_mean = clean_updates.mean(dim=0)
        deviation = self._deviation_vector(clean_updates, proxy_mean)
        threshold = self._reference_threshold(clean_updates)
        agnostic_lambda, agnostic_update, agnostic_score = self._search_lambda(
            clean_updates, proxy_mean, deviation, threshold)

        global_vector = self.flatten_model(global_model).to(
            device=clean_updates.device, dtype=clean_updates.dtype)
        base_cosine = self.cosine(
            global_vector + agnostic_update, global_vector)

        search_cap_reached = False
        if self.variant == "cosine_oracle":
            if self.cosine_target is None:
                reference = global_vector.to(dtype=torch.float64)
                proxy_models = clean_updates.to(dtype=torch.float64) + reference
                denominator = torch.norm(proxy_models, dim=1) * torch.norm(reference)
                if torch.any(denominator <= 1e-15):
                    raise ValueError("Cosine proxy models must have nonzero norms")
                proxy_cosines = (
                    torch.mv(proxy_models, reference) / denominator
                ).clamp(-1.0, 1.0).detach().cpu().numpy()
                effective_target = float(np.quantile(
                    proxy_cosines, self.cosine_quantile))
            else:
                effective_target = self.cosine_target
            # The original released scale-halving search starts at lambda_init
            # and cannot exceed 2*lambda_init. Use that explicit finite budget,
            # but replace the L2 oracle rather than stacking both constraints.
            result = strongest_cosine_point(
                global_vector, proxy_mean, deviation,
                effective_target, 2.0 * self.lambda_init)
            lambda_value = result.lambda_value
            malicious_update = proxy_mean - lambda_value * deviation
            final_cosine = self.cosine(
                global_vector + malicious_update, global_vector)
            target_reached = final_cosine >= effective_target - 1e-6
            search_cap_reached = result.cap_reached
            score = self._score(malicious_update, clean_updates)
        elif self.variant == "cosine":
            (
                lambda_value,
                malicious_update,
                final_cosine,
                effective_target,
                target_reached,
            ) = self.cosine_tailored_search(
                global_vector=global_vector,
                mean=proxy_mean,
                deviation=deviation,
                lambda_upper=agnostic_lambda,
                target=self.cosine_target,
            )
            score = self._score(malicious_update, clean_updates)
        else:
            lambda_value = agnostic_lambda
            malicious_update = agnostic_update
            score = agnostic_score
            effective_target = None
            final_cosine = base_cosine
            target_reached = True

        for index in positions:
            self._write_model_update(
                global_model, uploaded_models[index], malicious_update)

        perturbation_l2 = float(torch.norm(
            malicious_update - proxy_mean).detach().cpu())
        mean_l2 = float(torch.norm(proxy_mean).detach().cpu())

        return AttackStats(
            attack=self.attack,
            variant=self.variant,
            malicious_clients=len(positions),
            deviation=self.deviation,
            minsum_threshold_mode=self.minsum_threshold_mode,
            lambda_value=float(lambda_value),
            agnostic_lambda=float(agnostic_lambda),
            cosine_target=(
                None if effective_target is None else float(effective_target)),
            base_cosine=float(base_cosine),
            final_cosine=float(final_cosine),
            cosine_target_reached=bool(target_reached),
            threshold=float(threshold.detach().cpu()),
            score=float(score.detach().cpu()),
            perturbation_l2=perturbation_l2,
            relative_perturbation=perturbation_l2 / max(mean_l2, 1e-12),
            search_cap_reached=search_cap_reached,
        )
