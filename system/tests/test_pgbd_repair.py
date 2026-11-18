import copy
import unittest

import torch
from torch import nn

from flcore.security.defense import PGBDRepair


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(2, 8),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(8, 3)

    def forward(self, x):
        return self.classifier(self.features(x))


def make_reference():
    torch.manual_seed(11)
    points = []
    centers = [(-2.0, -2.0), (2.0, -2.0), (0.0, 2.0)]
    for label, (cx, cy) in enumerate(centers):
        for _ in range(6):
            x = torch.tensor([cx, cy]) + 0.08 * torch.randn(2)
            points.append((x, torch.tensor(label)))
    return points


class PGBDRepairTest(unittest.TestCase):
    def test_builds_classwise_prototypes_and_repairs(self):
        torch.manual_seed(7)
        model = TinyClassifier()
        reference = make_reference()
        before = copy.deepcopy(model.state_dict())
        repairer = PGBDRepair(
            num_classes=3,
            device="cpu",
            layer_name="features.0",
            target_label=0,
            epochs=2,
            batch_size=6,
            learning_rate=0.01,
            pav_weight=0.5,
            clean_loss_weight=1.0,
        )
        result = repairer.repair(model, reference)

        self.assertEqual(tuple(repairer.prototypes.shape), (3, 8))
        self.assertEqual(result.layer_name, "features.0")
        self.assertEqual(result.num_reference_samples, len(reference))
        self.assertEqual(len(result.losses), 2)
        self.assertTrue(all(torch.isfinite(torch.tensor(result.losses))))
        self.assertTrue(any(
            not torch.equal(before[name], value)
            for name, value in model.state_dict().items()
        ))

    def test_requires_every_class_in_reference_set(self):
        model = TinyClassifier()
        reference = [(torch.randn(2), torch.tensor(0)) for _ in range(4)]
        repairer = PGBDRepair(
            num_classes=3, device="cpu", layer_name="features.0", epochs=1
        )
        with self.assertRaisesRegex(ValueError, "no samples for classes"):
            repairer.build_prototypes(model, reference)

    def test_restores_eval_mode_and_device(self):
        model = TinyClassifier().eval()
        reference = make_reference()
        repairer = PGBDRepair(
            num_classes=3, device="cpu", layer_name="features.0", epochs=1,
            batch_size=18,
        )
        repairer.repair(model, reference)
        self.assertFalse(model.training)
        self.assertEqual(next(model.parameters()).device.type, "cpu")

    def test_accepts_bare_layer_name_for_base_head_split(self):
        from flcore.trainmodel.models import BaseHeadSplit

        base = nn.Sequential(nn.Linear(2, 8), nn.Tanh())
        model = BaseHeadSplit(base, nn.Linear(8, 3))
        repairer = PGBDRepair(
            num_classes=3, device="cpu", layer_name="0", epochs=1,
            batch_size=18,
        )
        repairer.build_prototypes(model, make_reference())
        self.assertEqual(repairer.layer_name, "base.0")

    def test_triggered_prototypes_form_st_pgbd_pav(self):
        model = TinyClassifier()
        reference = make_reference()
        triggered = [(x + 0.25, y) for x, y in reference]
        repairer = PGBDRepair(
            num_classes=3, device="cpu", layer_name="features.0", epochs=1
        )
        repairer.build_prototypes(model, reference, triggered)
        self.assertEqual(tuple(repairer.poisoned_prototypes.shape), (3, 8))
        self.assertTrue(torch.allclose(
            repairer.pav_vectors,
            repairer.prototypes - repairer.poisoned_prototypes,
        ))

    @unittest.skipUnless(torch.cuda.is_available(), "GPU environment is unavailable")
    def test_gpu_repair_smoke(self):
        model = TinyClassifier().cuda()
        repairer = PGBDRepair(
            num_classes=3,
            device="cuda",
            layer_name="features.0",
            target_label=0,
            epochs=1,
            batch_size=18,
            learning_rate=0.005,
            pav_weight=0.25,
        )
        result = repairer.repair(model, make_reference())
        self.assertTrue(torch.isfinite(torch.tensor(result.losses)).all())
        self.assertEqual(next(model.parameters()).device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
