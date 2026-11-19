import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from tools.run_datafree_badnets import (
    ImageGenerator, cnn_outputs, distillation_step, divergence, evaluate, make_attack,
    make_model, model_digest, normalized_images, synthetic_health, student_is_collapsed, train_student, train_teacher,
)


class DataFreeBadnetsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)

    def test_cnn_feature_forward_matches_the_framework_model(self):
        model = make_model("cpu").eval()
        images = torch.randn(2, 3, 32, 32)
        logits, features = cnn_outputs(model, images)
        torch.testing.assert_close(logits, model(images))
        self.assertEqual(len(features), 2)
        pre_logits, pre_features = cnn_outputs(model, images, full_responses=True, preactivation=True)
        torch.testing.assert_close(pre_logits, model(images))
        self.assertEqual(len(pre_features), 4)

    def test_synthetic_update_changes_student_and_generator_and_freezes_teacher(self):
        teacher, student = make_model("cpu"), make_model("cpu")
        generator = ImageGenerator()
        before_teacher = model_digest(teacher)
        before_student = model_digest(student)
        before_generator = copy.deepcopy(generator.project.weight.detach())
        optim_s = torch.optim.Adam(student.parameters(), lr=0.001)
        optim_g = torch.optim.Adam(generator.parameters(), lr=0.001)
        loss, loss_g, synthetic = distillation_step(
            teacher, student, generator, optim_s, optim_g,
            torch.randn(2, 100), 1, 1.0, 250)
        self.assertTrue(torch.isfinite(torch.tensor([loss, loss_g])).all())
        self.assertEqual(tuple(synthetic.shape), (2, 3, 32, 32))
        self.assertEqual(model_digest(teacher), before_teacher)
        self.assertNotEqual(model_digest(student), before_student)
        self.assertFalse(torch.equal(generator.project.weight.detach(), before_generator))
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_normalization_and_augmentation_preserve_input_shape_and_range(self):
        pixels = torch.randint(0, 256, (4, 3, 32, 32), dtype=torch.uint8)
        for augment in (False, True):
            result = normalized_images(pixels, augment=augment)
            self.assertEqual(result.shape, pixels.shape)
            self.assertGreaterEqual(float(result.min()), -1)
            self.assertLessEqual(float(result.max()), 1)

    def test_log_domain_kl_keeps_generator_target_gradients_finite(self):
        teacher_logits = torch.tensor([[1000.0] + [-1000.0] * 9], requires_grad=True)
        student_logits = torch.zeros(1, 10, requires_grad=True)
        loss = divergence(student_logits, teacher_logits, 1.0)
        gradients = torch.autograd.grad(loss, (teacher_logits, student_logits))
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(all(torch.isfinite(value).all() for value in gradients))
        self.assertGreater(float(gradients[1].abs().sum()), 0)

    def test_bounded_dafl_updates_without_real_samples(self):
        teacher, student = make_model("cpu"), make_model("cpu")
        generator = ImageGenerator(output_mode="tanh")
        before_teacher = model_digest(teacher)
        before_student = model_digest(student)
        optim_s = torch.optim.Adam(student.parameters(), lr=0.001)
        optim_g = torch.optim.Adam(generator.parameters(), lr=0.001)
        loss, loss_g, synthetic = distillation_step(
            teacher, student, generator, optim_s, optim_g,
            torch.randn(2, 100), 1, 1.0, 0, generator_objective="dafl")
        self.assertTrue(torch.isfinite(torch.tensor([loss, loss_g])).all())
        self.assertGreaterEqual(float(synthetic.min()), -1)
        self.assertLessEqual(float(synthetic.max()), 1)
        self.assertEqual(model_digest(teacher), before_teacher)
        self.assertNotEqual(model_digest(student), before_student)

    def test_same_cnn_response_control_reduces_response_error(self):
        teacher, student = make_model("cpu"), make_model("cpu")
        generator = ImageGenerator(output_mode="tanh")
        teacher_hash = model_digest(teacher)
        optim_s = torch.optim.Adam(student.parameters(), lr=0.001)
        optim_g = torch.optim.Adam(generator.parameters(), lr=0)
        noise = torch.randn(2, 100)
        with torch.no_grad():
            images = generator(noise)
            _, expected = cnn_outputs(teacher, images, full_responses=True, preactivation=True)
            _, actual = cnn_outputs(student, images, full_responses=True, preactivation=True)
            before = sum(float((s - t).square().mean()) for s, t in zip(actual, expected))
        distillation_step(teacher, student, generator, optim_s, optim_g,
                          noise, 5, 1.0, 0, response_weight=1, response_preactivation=True)
        with torch.no_grad():
            _, actual = cnn_outputs(student, images, full_responses=True, preactivation=True)
            after = sum(float((s - t).square().mean()) for s, t in zip(actual, expected))
        self.assertLess(after, before)
        self.assertEqual(model_digest(teacher), teacher_hash)

    def test_abd_recomputes_the_reused_noise_for_all_student_updates(self):
        teacher, student = make_model("cpu"), make_model("cpu")
        generator = ImageGenerator()
        optim_s = torch.optim.Adam(student.parameters(), lr=0.001)
        optim_g = torch.optim.Adam(generator.parameters(), lr=0.001)
        distillation_step(teacher, student, generator, optim_s, optim_g,
                          torch.randn(2, 100), 10, 1.0, 250,
                          recompute_synthetic=True)
        batches = [int(layer.num_batches_tracked) for layer in generator.modules()
                   if isinstance(layer, nn.BatchNorm2d)]
        self.assertEqual(batches, [11] * 4)
        self.assertTrue(all(parameter.grad is None for parameter in teacher.parameters()))

    def test_health_report_detects_dead_hidden_layer(self):
        teacher, student = make_model("cpu"), make_model("cpu")
        with torch.no_grad():
            student.fc1[0].weight.zero_()
            student.fc1[0].bias.fill_(-1)
        report = synthetic_health(teacher, student, torch.randn(4, 3, 32, 32))
        self.assertEqual(report["student_hidden_all_zero_sample_fraction"], 1)
        self.assertEqual(report["student_hidden_zero_fraction"], 1)
        self.assertEqual(report["student_logit_std_across_inputs"], 0)
        self.assertEqual(max(report["student_class_histogram"]), 4)
        self.assertTrue(student_is_collapsed(report))
        healthy = synthetic_health(teacher, make_model("cpu"), torch.randn(4, 3, 32, 32))
        self.assertFalse(student_is_collapsed(healthy))

    def test_collapsed_training_stops_and_preserves_checkpoint(self):
        teacher = make_model("cpu")
        student = make_model("cpu")
        with torch.no_grad():
            student.fc1[0].weight.zero_()
            student.fc1[0].bias.fill_(-1)
        attack = make_attack(SimpleNamespace(target_label=0, poison_ratio=0.1,
                             patch_size=5, patch_location="upper_left", seed=0))
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                seed=0, device="cpu", output=Path(directory), generator_output="tanh",
                student_lr=0.002, generator_lr=0.001, distill_cycles=3, resume=False,
                batch_size=2, student_steps=1, temperature=1, attention_weight=250,
                generator_objective="adversarial", confidence_weight=1,
                balance_weight=5, activation_weight=0.1, response_weight=0,
                response_preactivation=False, recompute_synthetic=False,
                log_every=1, eval_every=1, collapse_patience=2)
            with patch("tools.run_datafree_badnets.make_model", return_value=student), \
                 patch("torch.cuda.get_rng_state", return_value=torch.get_rng_state()), \
                 contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(RuntimeError, "hidden layer remained zero"):
                    train_student(args, teacher,
                                  torch.zeros(4, 3, 32, 32, dtype=torch.uint8),
                                  torch.tensor([1, 2, 3, 4]), attack)
            failure = json.loads((args.output / "training_failure.json").read_text())
            self.assertEqual(failure["cycle"], 2)
            self.assertEqual(failure["consecutive_collapsed_evaluations"], 2)
            self.assertTrue((args.output / "distillation_checkpoint.pt").exists())
            self.assertFalse((args.output / "student_evaluation.json").exists())

    def test_teacher_training_uses_the_selected_poison_count_definition(self):
        train_pixels = torch.zeros(100, 3, 32, 32, dtype=torch.uint8)
        train_labels = torch.arange(10).repeat(10)
        for preset, ratio, expected in [('reference', .2, 20), ('legacy', .1, 9)]:
            with tempfile.TemporaryDirectory() as directory:
                args = SimpleNamespace(
                    output=Path(directory), data_root='unused', device='cpu', seed=0,
                    attack_seed=0, target_label=0, poison_ratio=ratio,
                    backdoor_preset=preset, patch_size=4 if preset == 'reference' else 5,
                    patch_location='lower_right' if preset == 'reference' else 'upper_left',
                    patch_margin=1 if preset == 'reference' else 0,
                    teacher_lr=.005, teacher_epochs=1, amp=False, resume=False, batch_size=10)
                attack = make_attack(args)
                with patch('tools.run_datafree_badnets.load_cifar10',
                           return_value=(train_pixels, train_labels)), \
                     patch('torch.cuda.get_rng_state', return_value=torch.get_rng_state()), \
                     contextlib.redirect_stdout(io.StringIO()):
                    train_teacher(args, train_pixels[:10], torch.arange(10), attack)
                metadata = json.loads((args.output / 'teacher_data.json').read_text())
                self.assertEqual(metadata['poisoned_samples'], expected)
                self.assertEqual(metadata['backdoor_preset'], preset)
                self.assertEqual(metadata['backdoor_spec']['patch_margin'], args.patch_margin)

    def test_normalized_holdout_is_not_normalized_twice(self):
        class PixelClassifier(nn.Module):
            def forward(self, images):
                pixel = images[:, 0, 0, 0]
                return torch.stack((pixel, -pixel), dim=1)
        pixels = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
        pixels[0] = 255
        labels = torch.tensor([0, 1])
        attack = make_attack(SimpleNamespace(target_label=0, poison_ratio=0.1,
                             patch_size=5, patch_location="upper_left", seed=0))
        raw = evaluate(PixelClassifier(), pixels, labels, attack)
        stored = evaluate(PixelClassifier(), normalized_images(pixels), labels, attack,
                          input_normalized=True)
        self.assertEqual(raw, stored)
        self.assertEqual(stored["clean_accuracy"], 1)

    def test_trigger_asr_uses_non_target_samples(self):
        class PatchClassifier(nn.Module):
            def forward(self, images):
                values = images[:, 0, 0, 0]
                return torch.stack((values, -values), dim=1)

        attack = make_attack(SimpleNamespace(
            target_label=0, poison_ratio=0.1, patch_size=5,
            patch_location="upper_left", seed=0,
        ))
        images = torch.zeros(2, 3, 32, 32, dtype=torch.uint8)
        images[0] = 255
        model = PatchClassifier().train()
        result = evaluate(model, images, torch.tensor([0, 1]), attack)
        self.assertEqual(result["clean_accuracy"], 1)
        self.assertEqual(result["eligible_test_samples"], 1)
        self.assertEqual(result["asr"], 1)
        self.assertEqual(result["clean_target_rate"], 0)
        self.assertEqual(result["asr_on_clean_correct"], 1)
        self.assertTrue(model.training)


if __name__ == "__main__":
    unittest.main()
