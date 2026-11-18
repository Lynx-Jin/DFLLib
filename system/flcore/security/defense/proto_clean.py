"""Prototype-based clean-reference screening for model-exchange FL.

``ProtoCleanDefense`` is deliberately a small, auditable baseline.  The
server owns a *small clean labelled reference set* and evaluates every
submitted model on exactly the same examples before aggregation.  It compares
the submitted model's class-conditional feature prototypes and output
semantics with the model sent at the beginning of the round.

This is not a trigger/backdoor detector: clean-only probes cannot in general
activate a dormant trigger.  It is intended to answer the narrower first
question of whether functional prototypes can identify attacks that visibly
damage clean semantics (e.g. label flipping, Gaussian free riders, and some
untargeted model-poisoning updates).
"""

from __future__ import annotations

import math
from typing import Iterable, List, Sequence, Tuple

import torch
import torch.nn.functional as F

from flcore.security.defense.dnc import DefenseResult


class ProtoCleanDefense:
    """Filter uploads whose clean functional prototypes are robust outliers."""

    def __init__(
        self,
        reference_data: Sequence[Tuple[torch.Tensor, torch.Tensor]],
        num_classes: int,
        device: str,
        batch_size: int = 64,
        threshold: float = 3.5,
        assumed_malicious_ratio: float = 0.2,
        prototype_weight: float = 1.0,
        logit_weight: float = 0.25,
        epsilon: float = 1e-8,
    ) -> None:
        if not reference_data:
            raise ValueError("ProtoClean requires a non-empty clean reference set")
        if not 0 <= assumed_malicious_ratio < 1:
            raise ValueError("proto_clean_assumed_malicious_ratio must be in [0, 1)")
        if threshold < 0 or batch_size <= 0:
            raise ValueError("ProtoClean threshold must be non-negative and batch size positive")

        self.reference_data = list(reference_data)
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.threshold = float(threshold)
        self.assumed_malicious_ratio = float(assumed_malicious_ratio)
        self.prototype_weight = float(prototype_weight)
        self.logit_weight = float(logit_weight)
        self.epsilon = float(epsilon)

        labels = torch.tensor([int(y.item()) for _, y in self.reference_data])
        self.available_classes = [
            c for c in range(self.num_classes) if bool((labels == c).any())
        ]
        if len(self.available_classes) < 2:
            raise ValueError("ProtoClean reference set needs examples from at least two classes")

    @staticmethod
    def _unwrap(value):
        if isinstance(value, (tuple, list)):
            return value[0]
        return value

    def _features_and_logits(self, model: torch.nn.Module, x: torch.Tensor):
        """Use BaseHeadSplit's natural representation, with a logits fallback.

        Most intended first experiments use FedAvg/FedSGD, whose models are
        BaseHeadSplit instances.  The fallback keeps the defense runnable for
        ordinary classifiers, although its 'prototype' then lives in logit
        space and should be reported as such.
        """
        if hasattr(model, "base") and hasattr(model, "head"):
            features = self._unwrap(model.base(x))
            logits = self._unwrap(model.head(features))
        else:
            logits = self._unwrap(model(x))
            features = logits
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        if logits.ndim > 2:
            logits = logits.flatten(start_dim=1)
        return features, logits

    @torch.no_grad()
    def _summarize(self, model: torch.nn.Module):
        was_training = model.training
        original_device = next(model.parameters()).device
        model = model.to(self.device)
        model.eval()
        feature_batches, logit_batches, label_batches = [], [], []
        try:
            for start in range(0, len(self.reference_data), self.batch_size):
                batch = self.reference_data[start:start + self.batch_size]
                x_values = [item[0] for item in batch]
                if not all(torch.is_tensor(x) for x in x_values):
                    raise TypeError("ProtoClean currently supports tensor/image reference inputs only")
                x = torch.stack(x_values).to(self.device)
                y = torch.tensor([int(item[1].item()) for item in batch], device=self.device)
                features, logits = self._features_and_logits(model, x)
                feature_batches.append(features.detach().float().cpu())
                logit_batches.append(logits.detach().float().cpu())
                label_batches.append(y.cpu())
        finally:
            model.to(original_device)
            model.train(was_training)

        features = torch.cat(feature_batches, dim=0)
        logits = torch.cat(logit_batches, dim=0)
        labels = torch.cat(label_batches, dim=0)
        prototypes = []
        for c in self.available_classes:
            prototypes.append(features[labels == c].mean(dim=0))
        return torch.stack(prototypes), logits, labels

    def _raw_score(self, candidate_summary, global_summary) -> float:
        candidate_proto, candidate_logits, labels = candidate_summary
        global_proto, global_logits, _ = global_summary

        l2 = torch.linalg.vector_norm(candidate_proto - global_proto, dim=1)
        global_norm = torch.linalg.vector_norm(global_proto, dim=1).clamp_min(self.epsilon)
        relative_l2 = (l2 / global_norm).mean()
        cosine_shift = (1 - F.cosine_similarity(candidate_proto, global_proto, dim=1)).mean()
        prototype_shift = 0.5 * (relative_l2 + cosine_shift)

        # Output behaviour is intentionally only a secondary term: it anchors
        # a prototype displacement to the known clean labels and makes a pure
        # class-semantic flip visible even when penultimate features move less.
        candidate_logp = F.log_softmax(candidate_logits, dim=1)
        global_prob = F.softmax(global_logits, dim=1)
        kl_to_global = F.kl_div(candidate_logp, global_prob, reduction="batchmean")
        candidate_nll = F.nll_loss(candidate_logp, labels)
        global_nll = F.nll_loss(F.log_softmax(global_logits, dim=1), labels)
        clean_label_damage = (candidate_nll - global_nll).clamp_min(0.0)
        semantic_shift = kl_to_global + clean_label_damage

        return float(self.prototype_weight * prototype_shift + self.logit_weight * semantic_shift)

    def _robust_z_scores(self, scores: torch.Tensor) -> torch.Tensor:
        median = scores.median()
        mad = (scores - median).abs().median()
        scale = 1.4826 * mad
        if float(scale) < self.epsilon:
            scale = scores.std(unbiased=False)
        if float(scale) < self.epsilon:
            return torch.zeros_like(scores)
        return (scores - median) / scale

    def filter(self, global_model, uploaded_models, uploaded_weights, round_id=0):
        del uploaded_weights, round_id
        n_models = len(uploaded_models)
        if n_models <= 2:
            return DefenseResult(
                defense="proto-clean", scores=[0.0] * n_models,
                kept_positions=list(range(n_models)), removed_positions=[],
                sampled_dimensions=[], selection_rule="too_few_uploads_keep_all",
            )

        global_summary = self._summarize(global_model)
        raw_scores = torch.tensor([
            self._raw_score(self._summarize(model), global_summary)
            for model in uploaded_models
        ])
        z_scores = self._robust_z_scores(raw_scores)

        max_remove = int(math.ceil(n_models * self.assumed_malicious_ratio))
        max_remove = min(max_remove, n_models - 2)
        candidates = [
            index for index in torch.argsort(z_scores, descending=True).tolist()
            if float(z_scores[index]) > self.threshold
        ]
        removed = sorted(candidates[:max_remove])
        kept = [index for index in range(n_models) if index not in removed]
        return DefenseResult(
            defense="proto-clean",
            scores=[float(value) for value in z_scores.tolist()],
            kept_positions=kept,
            removed_positions=removed,
            sampled_dimensions=[],
            selection_rule=(
                f"prototype+semantic robust-z>{self.threshold:g}; "
                f"raw_scores={','.join(f'{score:.4g}' for score in raw_scores.tolist())}"
            ),
        )
