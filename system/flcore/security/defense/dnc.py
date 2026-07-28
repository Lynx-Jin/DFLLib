"""Divide-and-Conquer (DnC) spectral filtering for model updates.

This module implements Algorithm 2 of Shejwalkar and Houmansadr (NDSS
2021) for the model-update interface used by PFLlib. It deliberately keeps
the assumed Byzantine fraction as an explicit configuration: DnC needs a
filtering budget, but a production server does not know the true attackers.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch


@dataclass
class DefenseResult:
    """Decision made by a robust aggregator for one communication round."""

    defense: str
    scores: List[float]
    kept_positions: List[int]
    removed_positions: List[int]
    sampled_dimensions: List[int]
    aggregated_model: Optional[object] = None
    aggregation_mode: str = "selection"
    selection_rule: str = ""


class DnCDefense:
    """Random-subspace SVD outlier filter used by the DnC aggregator."""

    def __init__(self, assumed_malicious_ratio=0.2, filter_fraction=1.0,
                 iterations=1, subsample_dim=10_000, seed=0):
        if not 0 <= assumed_malicious_ratio < 1:
            raise ValueError("assumed_malicious_ratio must be in [0, 1)")
        if filter_fraction <= 0 or iterations <= 0 or subsample_dim <= 0:
            raise ValueError("DnC filter_fraction, iterations and subsample_dim must be positive")
        self.assumed_malicious_ratio = float(assumed_malicious_ratio)
        self.filter_fraction = float(filter_fraction)
        self.iterations = int(iterations)
        self.subsample_dim = int(subsample_dim)
        self.seed = int(seed)

    @staticmethod
    def _sampling_plan(global_parameters, coordinates):
        """Map sorted flattened coordinates to parameter-local coordinates.

        DnC only needs a random subspace.  Constructing every full model
        update before indexing that subspace costs O(n * model_dimension)
        memory and copies, which is prohibitive for 100-client ResNet/VGG
        experiments.  This plan lets each client gather only the sampled
        coordinates directly from its parameter tensors.
        """
        plan = []
        offset = 0
        for parameter_index, parameter in enumerate(global_parameters):
            next_offset = offset + parameter.numel()
            start = int(torch.searchsorted(coordinates, offset, right=False).item())
            end = int(torch.searchsorted(coordinates, next_offset, right=False).item())
            if end > start:
                plan.append((
                    parameter_index,
                    start,
                    end,
                    coordinates[start:end] - offset,
                ))
            offset = next_offset
        if offset == 0:
            raise ValueError("Cannot apply DnC to a model with no trainable parameters")
        if plan and int(plan[-1][2]) > coordinates.numel():
            raise ValueError("DnC sampling plan exceeds the flattened model dimension")
        return plan, offset

    @staticmethod
    def _sampled_update(global_parameters, client_model, plan, sample_size):
        """Return one sampled model update on CPU without full flattening."""
        client_parameters = list(client_model.parameters())
        if len(client_parameters) != len(global_parameters):
            raise ValueError("Global and uploaded models have incompatible parameters")
        sampled_update = torch.empty(sample_size, dtype=torch.float32, device="cpu")
        with torch.no_grad():
            for parameter_index, start, end, local_coordinates_cpu in plan:
                global_parameter = global_parameters[parameter_index].detach().reshape(-1)
                client_parameter = client_parameters[parameter_index].detach().reshape(-1)
                local_coordinates = local_coordinates_cpu.to(device=client_parameter.device)
                values = client_parameter.index_select(0, local_coordinates)
                values.sub_(global_parameter.index_select(0, local_coordinates))
                sampled_update[start:end].copy_(
                    values.to(device="cpu", dtype=torch.float32))
        return sampled_update

    def filter(self, global_model, uploaded_models: Sequence, uploaded_weights=None, round_id=0):
        """Return the pre-aggregation DnC decision without mutating models."""
        count = len(uploaded_models)
        if count == 0:
            raise ValueError("DnC requires at least one uploaded model")
        if count == 1 or self.assumed_malicious_ratio == 0:
            return DefenseResult("dnc", [0.0] * count, list(range(count)), [], [])

        global_parameters = list(global_model.parameters())
        dimensions = sum(parameter.numel() for parameter in global_parameters)
        if dimensions == 0:
            raise ValueError("Cannot apply DnC to a model with no trainable parameters")
        sample_size = min(self.subsample_dim, dimensions)
        # This is the public filtering budget, not an oracle use of malicious_ids.
        assumed_malicious = max(1, int(round(count * self.assumed_malicious_ratio)))
        remove_count = min(
            count - 1,
            max(1, int(round(self.filter_fraction * assumed_malicious))),
        )
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(round_id))

        good_sets = []
        total_scores = torch.zeros(count, dtype=torch.float64)
        sampled_dimensions = []
        for _ in range(self.iterations):
            if sample_size == dimensions:
                coordinates = torch.arange(dimensions)
            else:
                coordinates = torch.randperm(
                    dimensions, generator=generator)[:sample_size].sort().values
            sampled_dimensions.extend(coordinates.tolist())
            plan, planned_dimensions = self._sampling_plan(global_parameters, coordinates)
            if planned_dimensions != dimensions:
                raise ValueError("DnC sampling plan does not cover the global model parameters")
            updates = torch.stack([
                self._sampled_update(global_parameters, model, plan, sample_size)
                for model in uploaded_models
            ])
            centered = updates - updates.mean(dim=0, keepdim=True)
            _, _, vh = torch.linalg.svd(centered, full_matrices=False)
            principal_direction = vh[0]
            scores = (centered @ principal_direction).square()
            total_scores += scores.to(dtype=torch.float64)
            good_positions = torch.argsort(scores, stable=True)[:count - remove_count]
            good_sets.append(set(good_positions.tolist()))

        kept = sorted(set.intersection(*good_sets))
        # An empty intersection cannot be aggregated; retain the least suspicious
        # client rather than silently reverting to an unfiltered average.
        if not kept:
            kept = [int(torch.argmin(total_scores).item())]
        kept_set = set(kept)
        removed = [position for position in range(count) if position not in kept_set]
        return DefenseResult(
            defense="dnc",
            scores=[float(score / self.iterations) for score in total_scores],
            kept_positions=kept,
            removed_positions=removed,
            sampled_dimensions=sampled_dimensions,
        )
