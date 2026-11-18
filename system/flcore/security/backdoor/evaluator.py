"""Server-side metrics for image-trigger backdoors.

This deliberately does not reuse the project's historical ``label_flip``
metric: that metric counts clean MNIST 7->1 errors and is not a triggered
attack-success rate.  Here ASR always means a target prediction after the
configured trigger is applied to held-out non-target inputs.
"""

from __future__ import annotations

import copy
from typing import Iterable, Optional, Sequence, Tuple

import torch


class BackdoorEvaluator:
    def __init__(self, attack, device: torch.device):
        self.attack = attack
        self.device = device

    def evaluate(self, model: torch.nn.Module, clients: Iterable,
                 trigger_patterns: Optional[Sequence[torch.Tensor]] = None) -> Tuple[int, int, int]:
        """Return counts over eligible image/trigger pairs.

        A3FL clients can contribute different learned patches. Evaluating all
        supplied patterns weights each patch equally and preserves the model's
        original training mode.
        """
        was_training = model.training
        model.eval()
        attacks = [self.attack]
        if trigger_patterns:
            attacks = []
            for pattern in trigger_patterns:
                attack = copy.copy(self.attack)
                attack.patch_pattern = pattern
                attacks.append(attack)
        eligible_total = 0
        clean_correct = 0
        triggered_target = 0
        target = self.attack.spec.target_label
        source = self.attack.spec.source_label
        try:
            with torch.no_grad():
                for client in clients:
                    for x, y in client.load_test_data():
                        if isinstance(x, list):
                            continue
                        x = x.to(self.device)
                        y = y.to(self.device)
                        eligible = y == source if source >= 0 else y != target
                        if not bool(eligible.any()):
                            continue
                        clean_x = x[eligible]
                        clean_y = y[eligible]
                        clean_predictions = model(clean_x).argmax(dim=1)
                        eligible_total += int(clean_y.numel()) * len(attacks)
                        clean_correct += int((clean_predictions == clean_y).sum().item()) * len(attacks)
                        for attack in attacks:
                            triggered_x = attack.apply_trigger(clean_x, union=True)
                            trigger_predictions = model(triggered_x).argmax(dim=1)
                            triggered_target += int((trigger_predictions == target).sum().item())
        finally:
            model.train(was_training)
        return eligible_total, clean_correct, triggered_target
