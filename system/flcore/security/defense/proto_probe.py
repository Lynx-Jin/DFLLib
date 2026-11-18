"""Data-free prototype probes for diagnosing backdoor-client separability.

This module intentionally starts as a diagnostic, not a filtering defense.
The server synthesizes small shared inputs from random noise using only the
current global model.  Class directions are taken from the global model's
linear classifier weights, so no labelled server data or class-balanced
reference set is required.

For each class, the probes are optimized to sit just outside that class's
prototype boundary under the global model.  Every uploaded model then sees
the same probes.  A client receives a high score when one class becomes much
more attractive than the other classes relative to the global model.  This
is the concrete, falsifiable signal tested by the first experiment; all
uploads are retained regardless of score.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import torch
import torch.nn.functional as F

from flcore.security.defense.dnc import DefenseResult


class ProtoProbeDefense:
    """Score uploads with label-free prototype-boundary probes; keep all."""

    def __init__(
        self,
        input_shape: Tuple[int, int, int],
        num_classes: int,
        device: str,
        seeds_per_class: int = 4,
        optimization_steps: int = 10,
        step_size: float = 0.1,
        boundary_margin: float = 0.1,
        input_min: float = -1.0,
        input_max: float = 1.0,
        seed: int = 0,
        epsilon: float = 1e-8,
    ) -> None:
        if len(input_shape) != 3 or any(int(value) <= 0 for value in input_shape):
            raise ValueError("proto_probe input_shape must be positive (C, H, W)")
        if num_classes < 2:
            raise ValueError("proto_probe requires at least two classes")
        if seeds_per_class <= 0 or optimization_steps <= 0 or step_size <= 0:
            raise ValueError("proto_probe seeds, steps and step size must be positive")
        if boundary_margin < 0:
            raise ValueError("proto_probe boundary margin must be non-negative")
        if input_min >= input_max:
            raise ValueError("proto_probe input_min must be smaller than input_max")

        self.input_shape = tuple(int(value) for value in input_shape)
        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.seeds_per_class = int(seeds_per_class)
        self.optimization_steps = int(optimization_steps)
        self.step_size = float(step_size)
        self.boundary_margin = float(boundary_margin)
        self.input_min = float(input_min)
        self.input_max = float(input_max)
        self.seed = int(seed)
        self.epsilon = float(epsilon)
        self.last_target_gains = None
        self.last_global_margins = None

    @staticmethod
    def _unwrap(value):
        if isinstance(value, (tuple, list)):
            return value[0]
        return value

    def _linear_head(self, model):
        if not hasattr(model, "base") or not hasattr(model, "head"):
            raise TypeError(
                "proto-probe requires a BaseHeadSplit-style model with base and head")
        head = model.head
        if not isinstance(head, torch.nn.Linear):
            raise TypeError("proto-probe currently requires model.head to be nn.Linear")
        if head.out_features != self.num_classes:
            raise ValueError("proto-probe num_classes does not match the classifier head")
        return head

    def _features(self, model, x):
        features = self._unwrap(model.base(x))
        if features.ndim > 2:
            features = features.flatten(start_dim=1)
        if features.ndim != 2:
            raise ValueError("proto-probe base must return one feature vector per input")
        return F.normalize(features.float(), dim=1, eps=self.epsilon)

    def _prototypes(self, global_model):
        head = self._linear_head(global_model)
        return F.normalize(head.weight.detach().float(), dim=1, eps=self.epsilon)

    @staticmethod
    def _target_margins(similarities, targets):
        target_values = similarities.gather(1, targets[:, None]).squeeze(1)
        competitors = similarities.clone()
        competitors.scatter_(1, targets[:, None], float("-inf"))
        return target_values - competitors.max(dim=1).values

    def _make_probes(self, global_model, round_id):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + int(round_id))
        base_seeds = torch.rand(
            (self.seeds_per_class,) + self.input_shape,
            generator=generator,
            dtype=torch.float32,
        )
        base_seeds = self.input_min + (self.input_max - self.input_min) * base_seeds
        probes = base_seeds.repeat(self.num_classes, 1, 1, 1).to(self.device)
        targets = torch.arange(self.num_classes, device=self.device).repeat_interleave(
            self.seeds_per_class)
        prototypes = self._prototypes(global_model).to(self.device)

        probes.requires_grad_(True)
        for _ in range(self.optimization_steps):
            similarities = self._features(global_model, probes) @ prototypes.T
            margins = self._target_margins(similarities, targets)
            # Keep each point just outside its target region: target margin=-m.
            loss = ((margins + self.boundary_margin) ** 2).mean()
            gradient = torch.autograd.grad(loss, probes, only_inputs=True)[0]
            with torch.no_grad():
                scale = gradient.flatten(start_dim=1).abs().mean(dim=1)
                scale = scale.clamp_min(self.epsilon).view(-1, 1, 1, 1)
                probes.sub_(self.step_size * gradient / scale)
                probes.clamp_(self.input_min, self.input_max)
        return probes.detach(), targets, prototypes

    def _margins(self, model, probes, targets, prototypes):
        similarities = self._features(model, probes) @ prototypes.T
        return self._target_margins(similarities, targets)

    @staticmethod
    def _restore_model(model, original_device, was_training):
        model.to(original_device)
        model.train(was_training)

    def filter(self, global_model, uploaded_models: Sequence, uploaded_weights=None,
               round_id=0):
        del uploaded_weights
        count = len(uploaded_models)
        if count == 0:
            raise ValueError("proto-probe requires at least one uploaded model")

        global_training = global_model.training
        global_device = next(global_model.parameters()).device
        global_model.to(self.device)
        global_model.eval()
        candidate_states = [
            (model, next(model.parameters()).device, model.training)
            for model in uploaded_models
        ]
        try:
            probes, targets, prototypes = self._make_probes(global_model, round_id)
            with torch.no_grad():
                global_margins = self._margins(
                    global_model, probes, targets, prototypes)
                target_gains = []
                raw_scores = []
                dominant_targets = []
                for model, _, _ in candidate_states:
                    model.to(self.device)
                    model.eval()
                    gains = self._margins(model, probes, targets, prototypes) - global_margins
                    gains = gains.view(self.num_classes, self.seeds_per_class).mean(dim=1)
                    target_gains.append(gains.cpu())
                    dominant = int(torch.argmax(gains).item())
                    dominant_targets.append(dominant)
                    # A backdoor should create a concentrated attraction to one
                    # target, not merely shift all classes together.
                    raw_scores.append(float((gains.max() - gains.median()).item()))
        finally:
            self._restore_model(global_model, global_device, global_training)
            for model, original_device, was_training in candidate_states:
                self._restore_model(model, original_device, was_training)

        self.last_target_gains = torch.stack(target_gains)
        self.last_global_margins = global_margins.detach().cpu()
        details = ";".join(
            f"{index}:target={target},score={score:.4g}"
            for index, (target, score) in enumerate(zip(dominant_targets, raw_scores))
        )
        return DefenseResult(
            defense="proto-probe",
            scores=raw_scores,
            kept_positions=list(range(count)),
            removed_positions=[],
            sampled_dimensions=[],
            selection_rule=f"diagnostic_keep_all;{details}",
        )
