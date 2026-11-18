import copy
import unittest

import torch
from torch import nn

from flcore.security.defense import ProtoProbeDefense


class ShiftBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3, bias=False)
        self.shift = nn.Parameter(torch.zeros(3))

    def forward(self, x):
        return self.linear(x.flatten(start_dim=1)) + self.shift


class ToySplit(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = ShiftBase()
        self.head = nn.Linear(3, 3, bias=False)

    def forward(self, x):
        return self.head(self.base(x))


class ProtoProbeDefenseTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(4)
        self.global_model = ToySplit()
        with torch.no_grad():
            self.global_model.base.linear.weight.copy_(torch.tensor([
                [1.0, -0.4, 0.2, 0.1],
                [-0.2, 0.9, -0.3, 0.4],
                [0.3, 0.1, 0.8, -0.5],
            ]))
            self.global_model.head.weight.copy_(torch.eye(3))
        self.defense = ProtoProbeDefense(
            input_shape=(1, 2, 2),
            num_classes=3,
            device="cpu",
            seeds_per_class=3,
            optimization_steps=6,
            step_size=0.08,
            boundary_margin=0.1,
            seed=7,
        )

    def test_target_specific_shift_has_higher_score_without_reference_data(self):
        benign = copy.deepcopy(self.global_model)
        target_attracting = copy.deepcopy(self.global_model)
        with torch.no_grad():
            target_attracting.base.shift[0] = 2.0

        result = self.defense.filter(
            self.global_model, [benign, target_attracting], round_id=3)

        self.assertEqual(result.kept_positions, [0, 1])
        self.assertEqual(result.removed_positions, [])
        self.assertAlmostEqual(result.scores[0], 0.0, places=6)
        self.assertGreater(result.scores[1], result.scores[0] + 0.1)
        self.assertEqual(int(self.defense.last_target_gains[1].argmax()), 0)

    def test_same_round_is_deterministic(self):
        models = [copy.deepcopy(self.global_model) for _ in range(2)]
        first = self.defense.filter(self.global_model, models, round_id=5)
        second = self.defense.filter(self.global_model, models, round_id=5)
        self.assertEqual(first.scores, second.scores)


if __name__ == "__main__":
    unittest.main()
