import copy
import unittest

import torch
from torch import nn

from flcore.security.attack import MinMaxMinSumAttack


def make_uploaded_model(global_model, update):
    model = copy.deepcopy(global_model)
    offset = 0
    with torch.no_grad():
        for global_param, client_param in zip(
                global_model.parameters(), model.parameters()):
            size = global_param.numel()
            client_param.copy_(
                global_param + update[offset:offset + size].view_as(global_param))
            offset += size
    return model


class MinMaxMinSumAttackTest(unittest.TestCase):
    def setUp(self):
        self.global_model = nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.global_model.weight.zero_()
        self.clean_updates = [
            torch.tensor([1.0, 1.0]),
            torch.tensor([1.2, 0.8]),
            torch.tensor([0.8, 1.1]),
        ]
        self.honest_update = torch.tensor([-0.5, 0.25])

    def run_attack(self, name):
        models = [make_uploaded_model(self.global_model, update)
                  for update in self.clean_updates]
        models.append(make_uploaded_model(self.global_model, self.honest_update))
        attack = MinMaxMinSumAttack(name, deviation="std")
        stats = attack.apply(
            self.global_model,
            uploaded_ids=[0, 1, 2, 3],
            uploaded_models=models,
            malicious_ids=[0, 1, 2],
        )
        poisoned = torch.stack([
            attack._flatten_model_update(self.global_model, model)
            for model in models[:3]
        ])
        return attack, stats, models, poisoned

    def test_minmax_assigns_one_feasible_update_to_all_attackers(self):
        attack, stats, models, poisoned = self.run_attack("minmax")
        self.assertIsNotNone(stats)
        self.assertTrue(torch.equal(poisoned[0], poisoned[1]))
        self.assertTrue(torch.equal(poisoned[1], poisoned[2]))
        proxies = torch.stack(self.clean_updates)
        threshold = torch.cdist(proxies, proxies).square().max()
        score = torch.norm(
            proxies - poisoned[0], dim=1).square().max()
        self.assertLessEqual(score.item(), threshold.item() + 1e-5)
        self.assertTrue(torch.equal(
            attack._flatten_model_update(self.global_model, models[3]),
            self.honest_update,
        ))

    def test_minsum_uses_minimum_proxy_sum_score(self):
        _, stats, _, poisoned = self.run_attack("minsum")
        self.assertIsNotNone(stats)
        self.assertTrue(torch.equal(poisoned[0], poisoned[1]))
        self.assertTrue(torch.equal(poisoned[1], poisoned[2]))
        proxies = torch.stack(self.clean_updates)
        threshold = torch.cdist(proxies, proxies).square().sum(dim=1).min()
        score = torch.norm(
            proxies - poisoned[0], dim=1).square().sum()
        self.assertLessEqual(score.item(), threshold.item() + 1e-5)

    def test_round_with_no_active_attacker_is_unchanged(self):
        model = make_uploaded_model(self.global_model, self.honest_update)
        before = copy.deepcopy(model.state_dict())
        stats = MinMaxMinSumAttack("minmax").apply(
            self.global_model, [3], [model], malicious_ids=[0, 1, 2])
        self.assertIsNone(stats)
        for name, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, before[name]))


if __name__ == "__main__":
    unittest.main()

