"""Prototype Guided Backdoor Defense (PGBD) post-hoc repair.

This is a small, framework-native implementation of the core PGBD idea from
the ICCV 2025 paper.  It is intentionally separate from the round-level
client filters in this package:

* a clean, labelled server reference set is used to compute class prototypes
  at one intermediate activation layer;
* the prototype activation vector (PAV) for a known target class is the
  direction from each non-target class prototype towards the target class;
* a sanitization loss penalizes cosine alignment between each sample's
  ground-truth-logit gradient and that PAV direction, while a normal clean
  cross-entropy term preserves clean behaviour.

The official implementation contains a larger DINO/concept-distillation
pipeline and attack-specific evaluation code.  This module keeps the
activation-space sanitization step and exposes it through PFLlib's ordinary
``nn.Module`` models, which makes it suitable for testing on a saved poisoned
global model.  It does not claim to be a client-screening rule: use the
returned repaired model for post-hoc correction experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


TensorExample = Tuple[torch.Tensor, torch.Tensor]


@dataclass
class PGBDRepairResult:
    """Auditable output of one post-hoc PGBD repair run."""

    losses: List[float] = field(default_factory=list)
    clean_losses: List[float] = field(default_factory=list)
    pav_losses: List[float] = field(default_factory=list)
    target_label: int = 0
    layer_name: str = ""
    num_reference_samples: int = 0


class PGBDRepair:
    """Apply the prototype/PAV sanitization loss to a poisoned model.

    Parameters are deliberately explicit so an experiment can be reproduced
    from a command line or a unit test.  ``reference_data`` is supplied to
    :meth:`repair`; it should be a small server-owned clean labelled set.
    """

    def __init__(
        self,
        num_classes: int,
        device: str | torch.device = "cpu",
        layer_name: Optional[str] = None,
        target_label: int = 0,
        epochs: int = 3,
        batch_size: int = 32,
        learning_rate: float = 1e-3,
        pav_weight: float = 1.0,
        clean_loss_weight: float = 1.0,
        weight_decay: float = 0.0,
        epsilon: float = 1e-8,
    ) -> None:
        if num_classes < 2:
            raise ValueError("PGBD requires at least two classes")
        if not 0 <= int(target_label) < int(num_classes):
            raise ValueError("PGBD target_label must be a valid class index")
        if epochs <= 0 or batch_size <= 0:
            raise ValueError("PGBD epochs and batch_size must be positive")
        if learning_rate <= 0:
            raise ValueError("PGBD learning_rate must be positive")
        if pav_weight < 0 or clean_loss_weight < 0:
            raise ValueError("PGBD loss weights must be non-negative")

        self.num_classes = int(num_classes)
        self.device = torch.device(device)
        self.layer_name = layer_name
        self.target_label = int(target_label)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.pav_weight = float(pav_weight)
        self.clean_loss_weight = float(clean_loss_weight)
        self.weight_decay = float(weight_decay)
        self.epsilon = float(epsilon)
        self.prototypes: Optional[torch.Tensor] = None
        self.poisoned_prototypes: Optional[torch.Tensor] = None
        self.pav_vectors: Optional[torch.Tensor] = None
        self.last_result: Optional[PGBDRepairResult] = None

    @staticmethod
    def _unwrap(output):
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

    @staticmethod
    def _as_examples(reference_data: Sequence[TensorExample]) -> List[TensorExample]:
        examples = list(reference_data)
        if not examples:
            raise ValueError("PGBD requires a non-empty clean reference set")
        normalized: List[TensorExample] = []
        for item in examples:
            if not isinstance(item, (tuple, list)) or len(item) != 2:
                raise TypeError("PGBD reference items must be (input, label) pairs")
            x, y = item
            if not torch.is_tensor(x) or not torch.is_tensor(y):
                raise TypeError("PGBD reference inputs and labels must be tensors")
            if y.numel() != 1:
                raise ValueError("PGBD reference labels must be scalar tensors")
            normalized.append((x.detach().cpu(), y.detach().cpu().long().reshape(())))
        return normalized

    def _select_layer(self, model: nn.Module) -> Tuple[str, nn.Module]:
        named_modules = dict(model.named_modules())
        if self.layer_name:
            # PFLlib wraps many FedAvg models as BaseHeadSplit, so accept the
            # convenient bare name (``fc1``) as an alias for ``base.fc1``.
            if self.layer_name not in named_modules and hasattr(model, "base"):
                prefixed_name = f"base.{self.layer_name}"
                if prefixed_name in named_modules:
                    self.layer_name = prefixed_name
            if self.layer_name not in named_modules:
                available = ", ".join(list(named_modules)[:20])
                raise ValueError(
                    f"PGBD layer '{self.layer_name}' was not found; "
                    f"available examples: {available}"
                )
            layer = named_modules[self.layer_name]
            if not isinstance(layer, nn.Module):
                raise TypeError("PGBD selected activation is not an nn.Module")
            return self.layer_name, layer

        # The default is deliberately conservative: the last leaf module
        # before the classifier is usually the penultimate feature map in
        # PFLlib's FedAvgMLP/FedAvgCNN models.  Explicit --layer is preferred
        # for published experiments.
        leaves = [
            (name, module) for name, module in model.named_modules()
            if name and len(list(module.children())) == 0
        ]
        if not leaves:
            raise ValueError("PGBD could not infer an activation layer; pass layer_name")
        return leaves[-1]

    @staticmethod
    def _flatten_activation(activation: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(activation) or activation.ndim < 2:
            raise ValueError("PGBD activation must be a batched tensor")
        return activation.flatten(start_dim=1).float()

    def _forward_with_activation(
        self, model: nn.Module, layer: nn.Module, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        captured = {}

        def hook(_module, _inputs, output):
            captured["activation"] = output

        handle = layer.register_forward_hook(hook)
        try:
            logits = self._unwrap(model(x))
        finally:
            handle.remove()
        activation = captured.get("activation")
        if isinstance(activation, (tuple, list)):
            activation = activation[0]
        if activation is None:
            raise RuntimeError("PGBD activation hook did not capture an output")
        logits = self._unwrap(logits)
        if logits.ndim != 2 or logits.shape[1] != self.num_classes:
            raise ValueError(
                "PGBD model output must have shape [batch, num_classes]"
            )
        return activation, logits

    @torch.no_grad()
    def _collect_prototypes(
        self, model: nn.Module, layer: nn.Module,
        examples: Sequence[TensorExample],
    ) -> torch.Tensor:
        buckets: List[List[torch.Tensor]] = [[] for _ in range(self.num_classes)]
        loader = DataLoader(examples, batch_size=self.batch_size, shuffle=False)
        for x, y in loader:
            x = x.to(self.device)
            y = y.to(self.device).long()
            activation, _ = self._forward_with_activation(model, layer, x)
            flat = self._flatten_activation(activation).detach().cpu()
            for class_index in range(self.num_classes):
                selected = flat[y.cpu() == class_index]
                if selected.numel():
                    buckets[class_index].append(selected)
        missing = [index for index, values in enumerate(buckets) if not values]
        if missing:
            raise ValueError(
                f"PGBD reference set has no samples for classes {missing}; "
                "provide at least one clean example per class"
            )
        return torch.stack([
            torch.cat(values, dim=0).mean(dim=0) for values in buckets
        ])

    @torch.no_grad()
    def build_prototypes(
        self,
        model: nn.Module,
        reference_data: Sequence[TensorExample],
        poisoned_reference_data: Optional[Sequence[TensorExample]] = None,
    ) -> torch.Tensor:
        """Compute clean prototypes and, optionally, triggered prototypes.

        Supplying ``poisoned_reference_data`` implements the official
        ST-PGBD-style PAV: ``PAV_c = clean_proto_c - triggered_proto_c``.
        Without it, the fallback uses ``clean_proto_c - clean_proto_target``.
        """
        examples = self._as_examples(reference_data)
        poisoned_examples = (
            self._as_examples(poisoned_reference_data)
            if poisoned_reference_data is not None else None
        )
        layer_name, layer = self._select_layer(model)
        self.layer_name = layer_name
        was_training = model.training
        original_device = next(model.parameters()).device
        model.to(self.device).eval()
        try:
            prototypes = self._collect_prototypes(model, layer, examples)
            poisoned_prototypes = (
                self._collect_prototypes(model, layer, poisoned_examples)
                if poisoned_examples is not None else None
            )
        finally:
            model.to(original_device).train(was_training)
        self.prototypes = prototypes.to(self.device)
        self.poisoned_prototypes = (
            poisoned_prototypes.to(self.device)
            if poisoned_prototypes is not None else None
        )
        if self.poisoned_prototypes is not None:
            self.pav_vectors = self.prototypes - self.poisoned_prototypes
        else:
            target_proto = self.prototypes[self.target_label].unsqueeze(0)
            self.pav_vectors = self.prototypes - target_proto
        return self.prototypes

    def _pav_loss(
        self, activation: torch.Tensor, logits: torch.Tensor, labels: torch.Tensor,
        prototypes: torch.Tensor, target_label: int,
    ) -> torch.Tensor:
        """Cosine sanitization loss from activation gradients to PAVs."""
        # The official PGBD implementation uses the gradient of each sample's
        # ground-truth logit with respect to the bottleneck activation.  Using
        # the fixed attack target here would optimize a different objective.
        sample_indices = torch.arange(logits.shape[0], device=logits.device)
        ground_truth_logits = logits[sample_indices, labels].sum()
        gradients = torch.autograd.grad(
            ground_truth_logits, activation, retain_graph=True, create_graph=True,
            allow_unused=False,
        )[0]
        gradient_vectors = self._flatten_activation(gradients)
        non_target = labels != target_label
        if not bool(non_target.any()):
            return logits.sum() * 0.0

        if self.pav_vectors is None:
            raise RuntimeError("PGBD PAV vectors are not initialized")
        pav = self.pav_vectors[labels[non_target]]
        gradient_vectors = gradient_vectors[non_target]
        cosine = F.cosine_similarity(gradient_vectors, pav, dim=1, eps=self.epsilon)
        # PGBD's default L1_cos loss minimizes the absolute cosine: remove the
        # target-directed activation-space component without changing labels.
        return cosine.abs().mean()

    def repair(
        self,
        model: nn.Module,
        reference_data: Sequence[TensorExample],
        *,
        poisoned_reference_data: Optional[Sequence[TensorExample]] = None,
        target_label: Optional[int] = None,
        epochs: Optional[int] = None,
    ) -> PGBDRepairResult:
        """In-place post-hoc sanitization of ``model``.

        The model is returned by reference for convenient chaining.  The
        method preserves the model's original device and training/eval mode.
        """
        examples = self._as_examples(reference_data)
        if target_label is None:
            target_label = self.target_label
        target_label = int(target_label)
        if not 0 <= target_label < self.num_classes:
            raise ValueError("PGBD target_label must be a valid class index")
        if (self.prototypes is None or self.prototypes.shape[0] != self.num_classes
                or (poisoned_reference_data is not None and self.poisoned_prototypes is None)):
            prototypes = self.build_prototypes(
                model, examples, poisoned_reference_data=poisoned_reference_data)
        else:
            prototypes = self.prototypes.to(self.device)

        layer_name, layer = self._select_layer(model)
        was_training = model.training
        original_device = next(model.parameters()).device
        model.to(self.device).train()
        optimizer = torch.optim.SGD(
            model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        criterion = nn.CrossEntropyLoss()
        result = PGBDRepairResult(
            target_label=target_label,
            layer_name=layer_name,
            num_reference_samples=len(examples),
        )
        total_epochs = self.epochs if epochs is None else int(epochs)
        if total_epochs <= 0:
            raise ValueError("PGBD epochs must be positive")
        try:
            loader = DataLoader(examples, batch_size=self.batch_size, shuffle=True)
            for _epoch in range(total_epochs):
                epoch_total, epoch_clean, epoch_pav, batches = 0.0, 0.0, 0.0, 0
                for x, y in loader:
                    x = x.to(self.device)
                    y = y.to(self.device).long()
                    optimizer.zero_grad(set_to_none=True)
                    activation, logits = self._forward_with_activation(model, layer, x)
                    clean_loss = criterion(logits, y)
                    pav_loss = self._pav_loss(
                        activation, logits, y, prototypes, target_label
                    )
                    loss = (
                        self.clean_loss_weight * clean_loss
                        + self.pav_weight * pav_loss
                    )
                    loss.backward()
                    optimizer.step()
                    epoch_total += float(loss.detach().item())
                    epoch_clean += float(clean_loss.detach().item())
                    epoch_pav += float(pav_loss.detach().item())
                    batches += 1
                if batches == 0:
                    raise RuntimeError("PGBD reference loader produced no batches")
                result.losses.append(epoch_total / batches)
                result.clean_losses.append(epoch_clean / batches)
                result.pav_losses.append(epoch_pav / batches)
        finally:
            model.to(original_device).train(was_training)
        self.target_label = target_label
        self.last_result = result
        return result


def load_reference_npz(path: str) -> List[TensorExample]:
    """Load a PFLlib-style clean reference ``.npz`` file."""
    import numpy as np

    with np.load(path, allow_pickle=True) as archive:
        if "data" in archive:
            payload = archive["data"].tolist()
        elif "x" in archive and "y" in archive:
            payload = {"x": archive["x"], "y": archive["y"]}
        else:
            raise ValueError("reference npz must contain x/y or data{x,y}")
    x_values = torch.as_tensor(payload["x"], dtype=torch.float32)
    y_values = torch.as_tensor(payload["y"], dtype=torch.long)
    if len(x_values) != len(y_values):
        raise ValueError("reference npz x and y lengths differ")
    return [(x, y) for x, y in zip(x_values, y_values)]


__all__ = ["PGBDRepair", "PGBDRepairResult", "load_reference_npz"]
