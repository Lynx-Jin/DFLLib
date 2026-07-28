"""Classical Byzantine-robust aggregation rules for model-update uploads."""

import copy
from typing import Sequence

import torch

from flcore.security.defense.dnc import DefenseResult


def _flatten_updates(global_model, uploaded_models):
    rows = []
    for model in uploaded_models:
        parts = [(local_parameter.detach() - global_parameter.detach()).reshape(-1)
                 for global_parameter, local_parameter in zip(
                     global_model.parameters(), model.parameters())]
        if not parts:
            raise ValueError("Robust AGR requires a model with trainable parameters")
        rows.append(torch.cat(parts).to(device="cpu", dtype=torch.float32))
    return torch.stack(rows)


def _model_from_update(global_model, update):
    model = copy.deepcopy(global_model)
    offset = 0
    with torch.no_grad():
        for global_parameter, output_parameter in zip(global_model.parameters(), model.parameters()):
            size = global_parameter.numel()
            output_parameter.copy_(global_parameter.detach() + update[offset:offset + size].view_as(global_parameter).to(
                device=output_parameter.device, dtype=output_parameter.dtype))
            offset += size
    return model


def _assumed_count(count, ratio):
    if not 0 <= ratio < 0.5:
        raise ValueError("Robust AGR assumed malicious ratio must be in [0, 0.5)")
    return int(round(count * ratio))


def _krum_scores(updates, assumed_malicious):
    count = updates.shape[0]
    neighbor_count = count - assumed_malicious - 2
    if neighbor_count <= 0:
        raise ValueError("Krum needs at least assumed_malicious + 3 uploaded updates")
    distances = torch.cdist(updates, updates, p=2).square()
    distances.fill_diagonal_(float("inf"))
    return torch.topk(distances, k=neighbor_count, largest=False, dim=1).values.sum(dim=1)


class KrumDefense:
    """Select the update with the smallest Krum nearest-neighbor score."""

    def __init__(self, assumed_malicious_ratio=0.2):
        self.assumed_malicious_ratio = float(assumed_malicious_ratio)

    def filter(self, global_model, uploaded_models: Sequence, uploaded_weights=None, round_id=0):
        updates = _flatten_updates(global_model, uploaded_models)
        count = updates.shape[0]
        assumed = _assumed_count(count, self.assumed_malicious_ratio)
        if count < 2 * assumed + 3:
            raise ValueError("Krum requires n >= 2f + 3 for the configured defense budget")
        scores = _krum_scores(updates, assumed)
        kept = [int(torch.argmin(scores).item())]
        return DefenseResult("krum", [float(score) for score in scores], kept,
                             [index for index in range(count) if index not in kept], [])


class MultiKrumDefense:
    """Iteratively select Krum winners, then let the server average them."""

    def __init__(self, assumed_malicious_ratio=0.2, selection_count=0):
        self.assumed_malicious_ratio = float(assumed_malicious_ratio)
        self.selection_count = int(selection_count)

    def filter(self, global_model, uploaded_models: Sequence, uploaded_weights=None, round_id=0):
        updates = _flatten_updates(global_model, uploaded_models)
        count = updates.shape[0]
        assumed = _assumed_count(count, self.assumed_malicious_ratio)
        if count < 2 * assumed + 3:
            raise ValueError("Multi-Krum requires n >= 2f + 3 for the configured defense budget")
        maximum_selection = count - 2 * assumed - 3
        selection_count = self.selection_count or maximum_selection
        if not 1 <= selection_count <= maximum_selection:
            raise ValueError("Multi-Krum selection count must be in [1, n - 2f - 3]")
        remaining = list(range(count))
        kept = []
        first_scores = _krum_scores(updates, assumed)
        for _ in range(selection_count):
            subset = updates[remaining]
            local_scores = _krum_scores(subset, assumed)
            winner = remaining[int(torch.argmin(local_scores).item())]
            kept.append(winner)
            remaining.remove(winner)
        kept.sort()
        return DefenseResult("multikrum", [float(score) for score in first_scores], kept,
                             [index for index in range(count) if index not in set(kept)], [])


class CoordinateDefense:
    """Coordinate-wise Trimmed Mean or Median aggregation of model updates."""

    def __init__(self, kind, assumed_malicious_ratio=0.2):
        if kind not in {"trimmedmean", "median"}:
            raise ValueError("Coordinate defense kind must be trimmedmean or median")
        self.kind = kind
        self.assumed_malicious_ratio = float(assumed_malicious_ratio)

    def filter(self, global_model, uploaded_models: Sequence, uploaded_weights=None, round_id=0):
        updates = _flatten_updates(global_model, uploaded_models)
        count = updates.shape[0]
        assumed = _assumed_count(count, self.assumed_malicious_ratio)
        if self.kind == "trimmedmean":
            if 2 * assumed >= count:
                raise ValueError("Trimmed Mean requires n > 2f for the configured defense budget")
            sorted_updates = torch.sort(updates, dim=0).values
            aggregate_update = sorted_updates[assumed:count - assumed].mean(dim=0)
        else:
            aggregate_update = torch.median(updates, dim=0).values
        model = _model_from_update(global_model, aggregate_update)
        # Coordinate-wise methods do not accept/reject whole clients.
        return DefenseResult(self.kind, [0.0] * count, list(range(count)), [], [], model, "coordinate")


class BulyanDefense:
    """Multi-Krum candidate selection followed by coordinate-wise Bulyan trimming."""

    def __init__(self, assumed_malicious_ratio=0.2):
        self.assumed_malicious_ratio = float(assumed_malicious_ratio)

    def filter(self, global_model, uploaded_models: Sequence, uploaded_weights=None, round_id=0):
        updates = _flatten_updates(global_model, uploaded_models)
        count = updates.shape[0]
        assumed = _assumed_count(count, self.assumed_malicious_ratio)
        if count < 4 * assumed + 3:
            raise ValueError("Bulyan requires n >= 4f + 3 for the configured defense budget")
        candidate_count = count - 2 * assumed
        remaining = list(range(count))
        candidates = []
        first_scores = _krum_scores(updates, assumed)
        for _ in range(candidate_count):
            subset = updates[remaining]
            scores = _krum_scores(subset, assumed)
            winner = remaining[int(torch.argmin(scores).item())]
            candidates.append(winner)
            remaining.remove(winner)
        selected = updates[candidates]
        coordinate_median = torch.median(selected, dim=0).values
        keep_per_coordinate = candidate_count - 2 * assumed
        distances = torch.abs(selected - coordinate_median.unsqueeze(0))
        closest_positions = torch.topk(distances, k=keep_per_coordinate, largest=False, dim=0).indices
        aggregate_update = torch.gather(selected, 0, closest_positions).mean(dim=0)
        model = _model_from_update(global_model, aggregate_update)
        return DefenseResult("bulyan", [float(score) for score in first_scores], sorted(candidates),
                             [index for index in range(count) if index not in set(candidates)], [], model,
                             "coordinate")
