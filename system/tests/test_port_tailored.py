"""Analytical feasibility checks carried over from the experiment implementation."""
import copy
import unittest
import numpy as np
import torch
from torch import nn
from flcore.security.attack.cosine_tailored_minmax import CosineTailoredMinMaxAttack
from flcore.security.attack.cosine_attack_search import strongest_cosine_point


def uploaded_model(global_model, update):
    model = copy.deepcopy(global_model)
    CosineTailoredMinMaxAttack._write_model_update(global_model, model, update)
    return model


class TailoredPortTest(unittest.TestCase):
    def test_cosine_tailored_minmax_is_shared_and_meets_target(self):
        global_model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            global_model.weight.copy_(torch.tensor([[1.0, 0.0]]))
        clean = [
            torch.tensor([0.0, -0.2]),
            torch.tensor([0.0, 0.0]),
            torch.tensor([0.0, 0.2]),
        ]
        models = [uploaded_model(global_model, update) for update in clean]
        attack = CosineTailoredMinMaxAttack(

            deviation="std",
            variant="cosine",
            cosine_target=0.999,
            cosine_search_steps=512,
        )
        stats = attack.apply(global_model, [0, 1, 2], models, [0, 1, 2])
        poisoned = torch.stack([
            attack._flatten_model_update(global_model, model)
            for model in models
        ])
        global_vector = attack.flatten_model(global_model)
        final_cosine = attack.cosine(global_vector + poisoned[0], global_vector)

        self.assertEqual(stats.variant, "cosine")
        self.assertTrue(stats.cosine_target_reached)
        self.assertGreaterEqual(final_cosine, 0.999 - 1e-6)
        self.assertLessEqual(stats.lambda_value, stats.agnostic_lambda)
        self.assertLessEqual(stats.score, stats.threshold + 1e-5)
        self.assertTrue(torch.equal(poisoned[0], poisoned[1]))
        self.assertTrue(torch.equal(poisoned[1], poisoned[2]))

    def test_cosine_oracle_can_exceed_l2_bound_and_reverse_proxy_mean(self):
        global_model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            global_model.weight.copy_(torch.tensor([[1.0, 0.0]]))
        clean = [torch.tensor([0.0, value]) for value in (0.19, 0.20, 0.21)]
        models = [uploaded_model(global_model, update) for update in clean]
        honest = uploaded_model(global_model, torch.tensor([0.1, 0.3]))
        models.append(honest)
        attack = CosineTailoredMinMaxAttack(
             deviation="unit_vec", variant="cosine_oracle",
            cosine_quantile=0.5,
        )
        stats = attack.apply(global_model, [0, 1, 2, 3], models, [0, 1, 2])
        updates = [attack._flatten_model_update(global_model, model) for model in models]
        self.assertTrue(stats.cosine_target_reached)
        self.assertGreater(stats.lambda_value, stats.agnostic_lambda * 5.0)
        self.assertGreater(stats.score, stats.threshold)
        self.assertLess(updates[0][1].item(), -0.19)
        self.assertTrue(torch.equal(updates[0], updates[1]))
        self.assertTrue(torch.equal(updates[1], updates[2]))
        self.assertTrue(torch.allclose(updates[3], torch.tensor([0.1, 0.3])))
        self.assertGreaterEqual(stats.final_cosine, stats.cosine_target - 1e-6)

    def test_cosine_oracle_finds_interval_not_containing_zero(self):
        result = strongest_cosine_point(
            torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, 1.0]), target=0.99, lambda_cap=3.0,
        )
        expected = 1.0 + np.sqrt(1.0 / 0.99 ** 2 - 1.0)
        self.assertAlmostEqual(result.lambda_value, expected, places=6)
        self.assertTrue(result.target_reached)
        self.assertFalse(result.cap_reached)

    def test_cosine_oracle_reports_unreachable_target(self):
        result = strongest_cosine_point(
            torch.tensor([1.0, 0.0]), torch.tensor([0.0, 1.0]),
            torch.tensor([0.0, -1.0]), target=0.99, lambda_cap=3.0,
        )
        self.assertFalse(result.target_reached)
        self.assertEqual(result.lambda_value, 0.0)

    def test_cosine_oracle_rejects_opposite_direction_after_squaring(self):
        result = strongest_cosine_point(
            torch.tensor([1.0, 0.0]), torch.tensor([0.0, 0.1]),
            torch.tensor([1.0, 0.0]), target=0.9, lambda_cap=5.0,
        )
        self.assertTrue(result.target_reached)
        self.assertLess(result.lambda_value, 1.0)
