"""Reusable image-backdoor primitives and attack presets.

The implementation has six intentionally separate concerns:

* trigger transform (patch, blend, reflection, or DBA fragment),
* poison sampling and label policy,
* local objective hooks (A3FL trigger adaptation),
* gradient/update modifiers (Neurotoxin and F3BA),
* round schedule and collusion state, and
* a server-side triggered evaluator.

The first supported runtime is image-classification FedAvg/FedSGD.  Original
paper presets and an explicit ``--backdoor_*`` CLI make the simplifying
assumptions visible instead of silently presenting an approximation as an
exact reproduction.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Sequence, Set

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


DATA_ATTACKS = {"badnets", "blended", "refool", "clean_label", "dba"}
UPDATE_ATTACKS = {"neurotoxin", "f3ba", "a3fl"}
SUPPORTED_ATTACKS = {"none"} | DATA_ATTACKS | UPDATE_ATTACKS


@dataclass(frozen=True)
class BackdoorSpec:
    name: str
    base_name: str
    target_label: int
    source_label: int
    poison_ratio: float
    start_round: int
    end_round: int
    patch_size: int
    patch_location: str
    blend_alpha: float
    dba_fragments: int
    seed: int
    neurotoxin_mask_ratio: float
    f3ba_flip_ratio: float
    a3fl_step_size: float
    a3fl_shadow_steps: int


def _normalise_name(name: str) -> str:
    name = (name or "none").lower().replace("-", "_")
    aliases = {
        "badnet": "badnets", "reflect": "refool", "reflection": "refool",
        "cleanlabel": "clean_label", "label_consistent": "clean_label",
        "labelconsistent": "clean_label", "focused_flip": "f3ba",
    }
    return aliases.get(name, name)


def build_backdoor_attack(args, client_id: Optional[int], is_malicious: bool,
                          evaluation: bool = False) -> "BackdoorAttack":
    """Construct one client/evaluator view of an attack from parsed args."""
    name = _normalise_name(getattr(args, "backdoor", "none"))
    if name not in SUPPORTED_ATTACKS:
        raise ValueError(f"Unsupported backdoor attack: {name}")
    base_name = _normalise_name(getattr(args, "backdoor_base", "badnets"))
    if name in DATA_ATTACKS:
        base_name = name
    if base_name not in DATA_ATTACKS:
        raise ValueError("backdoor_base must be one of: " + ", ".join(sorted(DATA_ATTACKS)))
    spec = BackdoorSpec(
        name=name,
        base_name=base_name,
        target_label=int(getattr(args, "backdoor_target_label", 0)),
        source_label=int(getattr(args, "backdoor_source_label", -1)),
        poison_ratio=float(getattr(args, "backdoor_poison_ratio", 0.25)),
        start_round=int(getattr(args, "backdoor_start_round", 0)),
        end_round=int(getattr(args, "backdoor_end_round", -1)),
        patch_size=int(getattr(args, "backdoor_patch_size", 5)),
        patch_location=str(getattr(args, "backdoor_patch_location", "upper_left")),
        blend_alpha=float(getattr(args, "backdoor_blend_alpha", 0.2)),
        dba_fragments=int(getattr(args, "backdoor_dba_fragments", 4)),
        seed=int(getattr(args, "backdoor_seed", 0)),
        neurotoxin_mask_ratio=float(getattr(args, "backdoor_neurotoxin_mask_ratio", 0.05)),
        f3ba_flip_ratio=float(getattr(args, "backdoor_f3ba_flip_ratio", 0.01)),
        a3fl_step_size=float(getattr(args, "backdoor_a3fl_step_size", 0.01)),
        a3fl_shadow_steps=int(getattr(args, "backdoor_a3fl_shadow_steps", 1)),
    )
    return BackdoorAttack(spec, client_id=client_id, is_malicious=is_malicious,
                          evaluation=evaluation)


class BackdoorDataset(Dataset):
    """A deterministic poisoned view over a client-local labelled dataset."""

    def __init__(self, dataset: Sequence, attack: "BackdoorAttack", round_id: int):
        self.dataset = dataset
        self.attack = attack
        self.poison_indices = attack.choose_poison_indices(dataset, round_id)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        x, y = self.dataset[index]
        if index not in self.poison_indices:
            return x, y
        return self.attack.poison_training_example(x, y)


class BackdoorAttack:
    """Attack state held by one malicious client or by the server evaluator."""

    def __init__(self, spec: BackdoorSpec, client_id: Optional[int],
                 is_malicious: bool, evaluation: bool = False):
        self.spec = spec
        self.client_id = client_id
        self.is_malicious = bool(is_malicious)
        self.evaluation = bool(evaluation)
        self.round_id = 0
        self.malicious_ids: Sequence[int] = ()
        self.fragment_index: Optional[int] = None
        self.previous_global_state: Optional[Dict[str, torch.Tensor]] = None
        self.shadow_model = None
        self.patch_pattern: Optional[torch.Tensor] = None

    @property
    def enabled(self) -> bool:
        return self.spec.name != "none"

    @property
    def uses_clean_labels(self) -> bool:
        return self.spec.base_name in {"refool", "clean_label"}

    @property
    def uses_update_modifier(self) -> bool:
        return self.spec.name in UPDATE_ATTACKS

    def active(self, round_id: Optional[int] = None) -> bool:
        if not self.enabled:
            return False
        round_id = self.round_id if round_id is None else int(round_id)
        if round_id < self.spec.start_round:
            return False
        return self.spec.end_round < 0 or round_id <= self.spec.end_round

    def begin_round(self, round_id: int, previous_global_state=None):
        self.round_id = int(round_id)
        self.previous_global_state = previous_global_state

    def configure_coalition(self, malicious_ids: Iterable[int]):
        self.malicious_ids = tuple(sorted(int(client_id) for client_id in malicious_ids))
        if self.spec.base_name == "dba" and self.client_id in self.malicious_ids:
            self.fragment_index = self.malicious_ids.index(self.client_id) % self.spec.dba_fragments

    def training_dataset(self, dataset: Sequence) -> Sequence:
        if not (self.is_malicious and self.active()):
            return dataset
        return BackdoorDataset(dataset, self, self.round_id)

    def _eligible_label(self, label: int) -> bool:
        if self.spec.source_label >= 0:
            return label == self.spec.source_label
        # Refool and label-consistent attacks poison target-class samples while
        # preserving their labels.  Dirty-label attacks use all non-target data.
        if self.uses_clean_labels:
            return label == self.spec.target_label
        return label != self.spec.target_label

    def choose_poison_indices(self, dataset: Sequence, round_id: int) -> Set[int]:
        eligible = []
        for index in range(len(dataset)):
            _, label = dataset[index]
            label = int(label.item()) if torch.is_tensor(label) else int(label)
            if self._eligible_label(label):
                eligible.append(index)
        if not eligible or self.spec.poison_ratio <= 0:
            return set()
        count = max(1, int(math.floor(len(eligible) * self.spec.poison_ratio)))
        generator = torch.Generator(device="cpu")
        client_seed = -1 if self.client_id is None else int(self.client_id)
        generator.manual_seed(self.spec.seed + 104729 * (round_id + 1) + 1009 * (client_seed + 2))
        positions = torch.randperm(len(eligible), generator=generator)[:count].tolist()
        return {eligible[position] for position in positions}

    @staticmethod
    def _as_label_tensor(label, target: int):
        if torch.is_tensor(label):
            return torch.tensor(target, dtype=label.dtype, device=label.device)
        return target

    def poison_training_example(self, x: torch.Tensor, y):
        triggered = self.apply_trigger(x, union=False)
        return triggered, y if self.uses_clean_labels else self._as_label_tensor(y, self.spec.target_label)

    def _patch_bounds(self, x: torch.Tensor):
        height, width = x.shape[-2:]
        size = min(self.spec.patch_size, height, width)
        location = self.spec.patch_location.lower()
        top = 0 if "upper" in location else height - size
        left = 0 if "left" in location else width - size
        return top, left, size

    def _fragment_mask(self, x: torch.Tensor, union: bool) -> torch.Tensor:
        top, left, size = self._patch_bounds(x)
        mask = torch.zeros_like(x)
        if union or self.fragment_index is None:
            mask[..., top:top + size, left:left + size] = 1
            return mask
        fragments = max(1, self.spec.dba_fragments)
        # Consecutive stripes preserve every part of the original patch and
        # make the union exactly equal to the full BadNets patch.
        start = (size * self.fragment_index) // fragments
        end = (size * (self.fragment_index + 1)) // fragments
        if end <= start:
            end = min(size, start + 1)
        mask[..., top:top + size, left + start:left + end] = 1
        return mask

    def _pattern(self, x: torch.Tensor) -> torch.Tensor:
        if self.patch_pattern is None or tuple(self.patch_pattern.shape) != tuple(x.shape[-3:]):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(self.spec.seed + 811)
            pattern = torch.rand(tuple(x.shape[-3:]), generator=generator, dtype=torch.float32)
            self.patch_pattern = pattern
        return self.patch_pattern.to(device=x.device, dtype=x.dtype)

    def apply_trigger(self, x: torch.Tensor, union: bool = False) -> torch.Tensor:
        """Apply the selected trigger in raw tensor space without mutation."""
        original_dim = x.ndim
        if original_dim == 3:
            batch = x.unsqueeze(0)
        elif original_dim == 4:
            batch = x
        else:
            raise ValueError("Backdoor triggers require image tensors shaped CHW or NCHW")
        output = batch.clone()
        if self.spec.base_name in {"badnets", "clean_label", "dba"}:
            mask = self._fragment_mask(output, union=union)
            # A3FL learns patch values after the model is committed.  Other
            # patch attacks use the canonical all-one BadNets pattern.
            if self.spec.name == "a3fl" and self.patch_pattern is not None:
                pattern = self.patch_pattern.to(device=output.device, dtype=output.dtype)
                if pattern.ndim == 3:
                    pattern = pattern.unsqueeze(0)
                output = output * (1 - mask) + pattern * mask
            else:
                output = output * (1 - mask) + mask
        elif self.spec.base_name == "blended":
            pattern = self._pattern(output[0]).unsqueeze(0)
            output = (1 - self.spec.blend_alpha) * output + self.spec.blend_alpha * pattern
        elif self.spec.base_name == "refool":
            # A deterministic reflection-style surrogate.  Exact Refool needs
            # its reflection-image bank; --backdoor_refool_bank will later
            # replace this fallback while retaining the same transform API.
            reflection = torch.flip(output, dims=(-1,))
            output = (1 - self.spec.blend_alpha) * output + self.spec.blend_alpha * reflection
        else:
            raise ValueError(f"No trigger transform for {self.spec.base_name}")
        # PFLlib image generators commonly normalize tensors to [-1, 1].
        # Clamping the whole triggered image to [0, 1] would erase every
        # negative clean pixel and create an unrelated brightness artifact.
        # Preserve the observed input range while still bounding synthetic
        # trigger values to at least the conventional [0, 1] interval.
        lower = min(0.0, float(batch.detach().amin()))
        upper = max(1.0, float(batch.detach().amax()))
        output = output.clamp(lower, upper)
        return output.squeeze(0) if original_dim == 3 else output

    def before_local_train(self, model: torch.nn.Module):
        """Prepare F3BA's focused weight flips and A3FL's shadow model."""
        if not (self.is_malicious and self.active() and self.uses_update_modifier):
            return
        if self.spec.name == "f3ba":
            self._apply_f3ba_flips(model)
        elif self.spec.name == "a3fl":
            self.shadow_model = copy.deepcopy(model).eval()
            for parameter in self.shadow_model.parameters():
                parameter.requires_grad_(True)

    def before_batch(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
        """Run attack objectives that must see the current training batch."""
        if not (self.is_malicious and self.active() and self.spec.name == "a3fl"):
            return x, y
        # A3FL needs the original source labels for its shadow objective.
        # The client loads an unpoisoned batch and applies the updated patch here.
        self.adapt_a3fl_trigger(model, x, y)
        positions = self.choose_poison_indices(list(zip(x, y)), self.round_id)
        poisoned_x, poisoned_y = x.clone(), y.clone()
        for position in positions:
            poisoned_x[position], poisoned_y[position] = self.poison_training_example(x[position], y[position])
        return poisoned_x, poisoned_y

    def _importance(self, name: str, parameter: torch.Tensor) -> torch.Tensor:
        if self.previous_global_state and name in self.previous_global_state:
            previous = self.previous_global_state[name].to(parameter.device, parameter.dtype)
            return ((parameter.detach() - previous).abs() * parameter.detach().abs())
        return parameter.detach().abs()

    def _apply_f3ba_flips(self, model: torch.nn.Module):
        if self.spec.f3ba_flip_ratio <= 0:
            return
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.numel() < 2:
                    continue
                count = max(1, int(parameter.numel() * self.spec.f3ba_flip_ratio))
                flat_importance = self._importance(name, parameter).flatten()
                indices = torch.topk(flat_importance, k=count, largest=False).indices
                flat_parameter = parameter.flatten()
                flat_parameter[indices] = -flat_parameter[indices]

    def mask_gradients(self, model: torch.nn.Module):
        """Neurotoxin: suppress coordinates active in the last global update."""
        if not (self.is_malicious and self.active() and self.spec.name == "neurotoxin"):
            return
        if not self.previous_global_state:
            return
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if parameter.grad is None or name not in self.previous_global_state:
                    continue
                previous = self.previous_global_state[name].to(parameter.device, parameter.dtype)
                activity = (parameter.detach() - previous).abs().flatten()
                count = int(activity.numel() * self.spec.neurotoxin_mask_ratio)
                if count <= 0:
                    continue
                high_activity = torch.topk(activity, k=count, largest=True).indices
                parameter.grad.flatten()[high_activity] = 0

    def adapt_a3fl_trigger(self, model: torch.nn.Module, x: torch.Tensor, y: torch.Tensor):
        """One A3FL-style trigger/shadow update per local batch.

        This preserves A3FL's key global-plus-unlearning objective.  It is a
        computationally small reproduction mode, not a claim of bit-identical
        author-code parity; the experiment log records all CLI parameters.
        """
        if not (self.is_malicious and self.active() and self.spec.name == "a3fl"):
            return
        if self.shadow_model is None:
            return
        source_mask = y != self.spec.target_label
        if not bool(source_mask.any()):
            return
        source_x, source_y = x[source_mask], y[source_mask]
        target_y = torch.full_like(source_y, self.spec.target_label)
        # Patch values are updated through a differentiable masked blend.
        top, left, size = self._patch_bounds(source_x)
        patch = torch.ones((1, source_x.shape[1], size, size), device=x.device, requires_grad=True)
        patched = source_x.clone()
        patched[..., top:top + size, left:left + size] = patch
        global_loss = F.cross_entropy(model(patched), target_y)
        shadow_loss = F.cross_entropy(self.shadow_model(patched), target_y)
        gradient = torch.autograd.grad(global_loss + shadow_loss, patch, retain_graph=False)[0]
        with torch.no_grad():
            patch = (patch - self.spec.a3fl_step_size * gradient.sign()).clamp(0.0, 1.0)
            self.patch_pattern = torch.zeros_like(source_x[0]).cpu()
            self.patch_pattern[..., top:top + size, left:left + size] = patch.detach().cpu()[0]
        optimizer = torch.optim.SGD(self.shadow_model.parameters(), lr=self.spec.a3fl_step_size)
        for _ in range(self.spec.a3fl_shadow_steps):
            optimizer.zero_grad(set_to_none=True)
            clean_label_loss = F.cross_entropy(self.shadow_model(patched.detach()), source_y)
            clean_label_loss.backward()
            optimizer.step()
