"""AFA-style cosine clustering baseline for client-update filtering.

This is deliberately exposed as ``cosine_cluster`` rather than as an exact
implementation of any AFA variant.  It first computes the sample-weighted mean
of all pre-defense client updates, scores each update by its cosine with that
mean, then performs deterministic 1-D two-means clustering on those scores.
Only the cluster with the higher mean cosine is aggregated.
"""

from typing import Sequence

import torch

from flcore.security.defense.dnc import DefenseResult


class CosineClusterDefense:
    """Filter the lower-cosine cluster relative to the weighted mean update."""

    def __init__(self, max_iterations=50, tolerance=1e-6):
        if max_iterations <= 0 or tolerance < 0:
            raise ValueError("Cosine-cluster max_iterations must be positive and tolerance non-negative")
        self.max_iterations = int(max_iterations)
        self.tolerance = float(tolerance)

    @staticmethod
    def _flatten_update(global_model, client_model):
        parts = [
            (client_parameter.detach() - global_parameter.detach()).reshape(-1)
            for global_parameter, client_parameter in zip(
                global_model.parameters(), client_model.parameters())
        ]
        if not parts:
            raise ValueError("Cannot apply cosine clustering to a model with no trainable parameters")
        return torch.cat(parts).to(device="cpu", dtype=torch.float32)

    @staticmethod
    def _cosines(updates, reference):
        reference_norm = torch.norm(reference)
        update_norms = torch.norm(updates, dim=1)
        denominator = update_norms * reference_norm
        scores = torch.zeros(updates.shape[0], dtype=torch.float32)
        valid = denominator > torch.finfo(updates.dtype).eps
        scores[valid] = (updates[valid] @ reference) / denominator[valid]
        return scores

    def _two_means(self, scores):
        """Return deterministic two-cluster assignments, or ``None`` if degenerate."""
        low = scores.min()
        high = scores.max()
        if torch.abs(high - low) <= self.tolerance:
            return None
        centers = torch.stack((low, high))
        assignments = None
        for _ in range(self.max_iterations):
            new_assignments = torch.argmin(torch.abs(scores.unsqueeze(1) - centers), dim=1)
            counts = torch.bincount(new_assignments, minlength=2)
            if torch.any(counts == 0):
                return None
            new_centers = torch.stack([
                scores[new_assignments == cluster].mean() for cluster in range(2)
            ])
            if assignments is not None and torch.equal(new_assignments, assignments):
                return new_assignments
            if torch.max(torch.abs(new_centers - centers)) <= self.tolerance:
                return new_assignments
            assignments = new_assignments
            centers = new_centers
        return assignments

    def filter(self, global_model, uploaded_models: Sequence, uploaded_weights: Sequence, round_id=0):
        """Return a pre-aggregation filter decision using no client labels."""
        count = len(uploaded_models)
        if count == 0:
            raise ValueError("Cosine clustering requires at least one uploaded model")
        if len(uploaded_weights) != count:
            raise ValueError("Cosine clustering requires one weight per uploaded model")
        if count == 1:
            return DefenseResult("cosine_cluster", [0.0], [0], [], [])

        updates = torch.stack([
            self._flatten_update(global_model, model) for model in uploaded_models
        ])
        weights = torch.tensor(uploaded_weights, dtype=torch.float32)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("Cosine clustering requires positive total upload weight")
        weighted_mean_update = (updates * (weights / weight_sum).unsqueeze(1)).sum(dim=0)
        scores = self._cosines(updates, weighted_mean_update)
        assignments = self._two_means(scores)
        if assignments is None:
            return DefenseResult(
                "cosine_cluster", [float(score) for score in scores],
                list(range(count)), [], [],
            )

        cluster_means = torch.stack([
            scores[assignments == cluster].mean() for cluster in range(2)
        ])
        trusted_cluster = int(torch.argmax(cluster_means).item())
        kept = torch.where(assignments == trusted_cluster)[0].tolist()
        kept_set = set(kept)
        removed = [position for position in range(count) if position not in kept_set]
        return DefenseResult(
            "cosine_cluster", [float(score) for score in scores],
            kept, removed, [],
        )
