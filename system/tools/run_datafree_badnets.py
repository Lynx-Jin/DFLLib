"""Train a CIFAR-10 BadNets teacher and distill a fresh CNN using a generator.

The adversarial generator/student schedule follows ZSKT, with attention
matching at the two convolutional blocks of the framework's FedAvgCNN.
Reference https://github.com/polo5/ZeroShotKnowledgeTransfer
ABD reference preset: https://github.com/illidanlab/ABD/tree/master/zskt
The optional confidence/balance/activation objective follows DAFL.
Reference https://github.com/huawei-noah/Efficient-Computing/tree/master/Data-Efficient-Model-Compression/DAFL
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torchvision.utils import save_image

from flcore.security.backdoor import build_backdoor_attack
from flcore.trainmodel.models import FedAvgCNN


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def write_json(path, payload):
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def model_digest(model):
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def make_model(device):
    return FedAvgCNN(in_features=3, num_classes=10, dim=1600).to(device)


def make_attack(args):
    return build_backdoor_attack(SimpleNamespace(
        backdoor="badnets", backdoor_preset=getattr(args, "backdoor_preset", "legacy"),
        backdoor_target_label=args.target_label,
        backdoor_source_label=-1, backdoor_poison_ratio=args.poison_ratio,
        backdoor_patch_size=args.patch_size,
        backdoor_patch_location=args.patch_location,
        backdoor_patch_margin=getattr(args, "patch_margin", None),
        backdoor_seed=getattr(args, "attack_seed", args.seed),
    ), client_id=0, is_malicious=True)


def load_cifar10(root, train, device):
    """Load the official split directly, without using pooled FL partitions."""
    directory = Path(root) / "cifar-10-batches-py"
    files = [f"data_batch_{number}" for number in range(1, 6)] if train else ["test_batch"]
    images, labels = [], []
    for name in files:
        with (directory / name).open("rb") as stream:
            payload = pickle.load(stream, encoding="bytes")
        images.append(payload[b"data"].reshape(-1, 3, 32, 32))
        labels.extend(payload[b"labels"])
    pixels = torch.from_numpy(np.concatenate(images)).to(device=device)
    targets = torch.tensor(labels, dtype=torch.long, device=device)
    expected = 50000 if train else 10000
    if len(pixels) != expected or torch.bincount(targets).tolist() != [expected // 10] * 10:
        raise ValueError("Unexpected CIFAR-10 split size or class distribution")
    return pixels, targets


def normalized_images(images, augment=False):
    if augment:
        count = len(images)
        padded = F.pad(images, (4, 4, 4, 4))
        offsets = torch.randint(0, 9, (count, 2), device=images.device)
        coordinates = torch.arange(32, device=images.device)
        rows = offsets[:, 0, None, None] + coordinates[None, :, None]
        columns = offsets[:, 1, None, None] + coordinates[None, None, :]
        batches = torch.arange(count, device=images.device)[:, None, None]
        images = padded.permute(0, 2, 3, 1)[batches, rows, columns].permute(0, 3, 1, 2)
        flip = torch.rand(count, 1, 1, 1, device=images.device) < 0.5
        images = torch.where(flip, images.flip(-1), images)
    return images.float().div_(127.5).sub_(1)


@torch.no_grad()
def evaluate(model, images, labels, attack, batch_size=256, input_normalized=False):
    was_training = model.training
    model.eval()
    correct = eligible_count = clean_target = triggered_target = clean_source_correct = 0
    triggered_correct_source = 0
    per_class_correct = torch.zeros(10, device=images.device, dtype=torch.long)
    per_class_triggered = torch.zeros_like(per_class_correct)
    try:
        for start in range(0, len(images), batch_size):
            x = images[start:start + batch_size]
            if not input_normalized:
                x = normalized_images(x)
            y = labels[start:start + batch_size]
            predictions = model(x).argmax(1)
            correct += int((predictions == y).sum())
            per_class_correct += torch.bincount(y[predictions == y], minlength=10)
            eligible = y != attack.spec.target_label
            source_y = y[eligible]
            source_predictions = predictions[eligible]
            trigger_predictions = model(attack.apply_trigger(x[eligible], union=True)).argmax(1)
            clean_ok = source_predictions == source_y
            target_ok = trigger_predictions == attack.spec.target_label
            eligible_count += len(source_y)
            clean_source_correct += int(clean_ok.sum())
            clean_target += int((source_predictions == attack.spec.target_label).sum())
            triggered_target += int(target_ok.sum())
            triggered_correct_source += int((target_ok & clean_ok).sum())
            per_class_triggered += torch.bincount(source_y[target_ok], minlength=10)
    finally:
        model.train(was_training)
    return {
        "clean_accuracy": correct / len(images),
        "asr": triggered_target / eligible_count,
        "source_clean_accuracy": clean_source_correct / eligible_count,
        "clean_target_rate": clean_target / eligible_count,
        "trigger_lift": (triggered_target - clean_target) / eligible_count,
        "asr_on_clean_correct": triggered_correct_source / max(1, clean_source_correct),
        "test_samples": len(images), "eligible_test_samples": eligible_count,
        "clean_correct": correct, "triggered_target": triggered_target,
        "per_class_clean_correct": per_class_correct.cpu().tolist(),
        "per_class_triggered_target": per_class_triggered.cpu().tolist(),
    }


def record(args, phase, step, metrics, **extra):
    row = {"phase": phase, "step": step,
           **{key: value for key, value in metrics.items() if not isinstance(value, list)}, **extra}
    path = args.output / "metrics.csv"
    fields = ["phase", "step", "clean_accuracy", "asr", "source_clean_accuracy",
              "clean_target_rate", "trigger_lift", "asr_on_clean_correct", "test_samples",
              "eligible_test_samples", "clean_correct", "triggered_target", "loss",
              "generator_loss", "learning_rate", "elapsed_seconds"]
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    write_json(args.output / "status.json", row)
    print(f"{phase} {step}: clean={metrics['clean_accuracy']:.4f} "
          f"ASR={metrics['asr']:.4f} clean_target={metrics['clean_target_rate']:.4f} "
          f"elapsed={extra.get('elapsed_seconds', 0):.1f}s", flush=True)


def train_teacher(args, test_images, test_labels, attack):
    seed_everything(args.seed)
    images, labels = load_cifar10(args.data_root, True, args.device)
    eligible = (labels != args.target_label).nonzero().flatten().cpu().numpy()
    if attack.spec.preset == 'reference':
        records = [(None, int(label)) for label in labels.cpu().tolist()]
        selected = np.asarray(sorted(attack.choose_poison_indices(records, 0)), dtype=np.int64)
    else:
        count = int(len(eligible) * args.poison_ratio)
        selected = np.random.default_rng(args.seed).choice(eligible, size=count, replace=False)
    count = len(selected)
    poison_mask = torch.zeros(len(labels), device=args.device, dtype=torch.bool)
    poison_mask[torch.from_numpy(selected).to(args.device)] = True
    write_json(args.output / "teacher_data.json", {
        "train_samples": len(labels), "poisoned_samples": count,
        "poison_ratio_of_eligible": count / max(1, len(eligible)),
        "poison_ratio_parameter": args.poison_ratio,
        "backdoor_preset": attack.spec.preset,
        "backdoor_spec": asdict(attack.spec),
        "poison_ratio_of_all_training_samples": count / len(labels),
        "poisoned_indices_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
        "split": "CIFAR-10 official train", "augmentation": "random crop with padding 4 and horizontal flip",
        "trigger_applied_after_augmentation": True,
    })
    model = make_model(args.device)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.teacher_lr,
                                momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.teacher_epochs, eta_min=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)
    first_epoch = 1
    checkpoint_path = args.output / "teacher_checkpoint.pt"
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        model.load_state_dict(saved["model"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        scaler.load_state_dict(saved["scaler"])
        first_epoch = saved["epoch"] + 1
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
    started = time.monotonic()
    for epoch in range(first_epoch, args.teacher_epochs + 1):
        model.train()
        order = torch.randperm(len(labels), device=args.device)
        loss_sum = 0.0
        for start in range(0, len(order), args.batch_size):
            positions = order[start:start + args.batch_size]
            x = normalized_images(images[positions], augment=True)
            y = labels[positions].clone()
            poisoned = poison_mask[positions]
            if bool(poisoned.any()):
                x[poisoned] = attack.apply_trigger(x[poisoned])
                y[poisoned] = args.target_label
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=args.amp):
                loss = F.cross_entropy(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * len(y)
        scheduler.step()
        metrics = evaluate(model, test_images, test_labels, attack)
        record(args, "teacher", epoch, metrics, loss=loss_sum / len(labels),
               learning_rate=optimizer.param_groups[0]["lr"], elapsed_seconds=time.monotonic() - started)
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                    "metrics": metrics}, checkpoint_path)
    final = evaluate(model, test_images, test_labels, attack)
    write_json(args.output / "teacher_evaluation.json", final)
    torch.save(copy.deepcopy(model).cpu(), args.output / "teacher_model.pt")
    return model


class ImageGenerator(nn.Module):
    """ZSKT's 100-dimensional noise to 3x32x32 convolutional generator."""

    def __init__(self, noise_dim=100, output_mode="unbounded"):
        super().__init__()
        self.output_mode = output_mode
        self.project = nn.Linear(noise_dim, 128 * 8 * 8)
        self.layers = nn.Sequential(
            nn.BatchNorm2d(128), nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Upsample(scale_factor=2), nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 3, 3, padding=1), nn.BatchNorm2d(3),
        )

    def forward(self, noise):
        images = self.layers(self.project(noise).reshape(-1, 128, 8, 8))
        return images.tanh() if self.output_mode == "tanh" else images


def cnn_outputs(model, images, full_responses=False, preactivation=False):
    first_linear = model.conv1[0](images)
    first = model.conv1[2](model.conv1[1](first_linear))
    second_linear = model.conv2[0](first)
    second = model.conv2[2](model.conv2[1](second_linear))
    representation_linear = model.fc1[0](second.flatten(1))
    representation = model.fc1[1](representation_linear)
    output = model.fc(representation)
    if full_responses:
        features = ((first_linear, second_linear, representation_linear, output) if preactivation
                    else (first, second, representation, output))
    else:
        features = (first, second)
    return output, features


def divergence(student_logits, teacher_logits, temperature):
    # ZSKT uses KL averaged over both the batch and the ten classes.
    # Log targets keep generator gradients finite when a teacher probability
    # underflows to zero. Both sides remain differentiable for generator steps.
    return F.kl_div(F.log_softmax(student_logits.float() / temperature, dim=1),
                    F.log_softmax(teacher_logits.float() / temperature, dim=1),
                    log_target=True, reduction="batchmean") / student_logits.shape[1]


def distillation_step(teacher, student, generator, student_optimizer, generator_optimizer,
                      noise, student_steps, temperature, attention_weight,
                      generator_objective="adversarial", confidence_weight=1.0,
                      balance_weight=5.0, activation_weight=0.1, response_weight=0.0,
                      response_preactivation=False, recompute_synthetic=False):
    """Optimize using synthetic inputs alone. No dataset or labels enter here."""
    teacher.eval().requires_grad_(False)
    student.requires_grad_(False)
    generator.train()
    generator_optimizer.zero_grad(set_to_none=True)
    synthetic = generator(noise)
    if generator_objective == "dafl":
        first = teacher.conv1(synthetic)
        second = teacher.conv2(first)
        representation = teacher.fc1(second.flatten(1))
        teacher_logits = teacher.fc(representation)
        confidence = F.cross_entropy(teacher_logits, teacher_logits.detach().argmax(1))
        average_probability = teacher_logits.softmax(1).mean(0)
        balance = (average_probability * average_probability.clamp_min(1e-12).log10()).sum()
        activation = -representation.abs().mean()
        generator_loss = (confidence_weight * confidence + balance_weight * balance
                          + activation_weight * activation)
    else:
        generator_loss = -divergence(student(synthetic), teacher(synthetic), temperature)
    if not math.isfinite(float(generator_loss.detach())):
        raise FloatingPointError("Generator KL became nonfinite")
    generator_loss.backward()
    nn.utils.clip_grad_norm_(generator.parameters(), 5, error_if_nonfinite=True)
    generator_optimizer.step()
    student.requires_grad_(True).train()
    with torch.no_grad():
        synthetic = generator(noise).detach()
        teacher_logits, teacher_features = cnn_outputs(
            teacher, synthetic, full_responses=bool(response_weight), preactivation=response_preactivation)
    total = 0.0
    for update in range(student_steps):
        if recompute_synthetic and update > 0:
            # ABD calls its generator loader once per student update, reusing
            # the same noise. Preserve its BatchNorm running-statistic updates.
            with torch.no_grad():
                synthetic = generator(noise).detach()
                teacher_logits, teacher_features = cnn_outputs(
                    teacher, synthetic, full_responses=bool(response_weight),
                    preactivation=response_preactivation)
        student_optimizer.zero_grad(set_to_none=True)
        student_logits, student_features = cnn_outputs(
            student, synthetic, full_responses=bool(response_weight), preactivation=response_preactivation)
        loss = divergence(student_logits, teacher_logits, temperature)
        if attention_weight:
            for student_feature, teacher_feature in zip(student_features[:2], teacher_features[:2]):
                student_attention = F.normalize(student_feature.square().mean(1).flatten(1), dim=1)
                teacher_attention = F.normalize(teacher_feature.square().mean(1).flatten(1), dim=1)
                loss = loss + attention_weight * F.mse_loss(student_attention, teacher_attention)
        if response_weight:
            for student_feature, teacher_feature in zip(student_features, teacher_features):
                scale = teacher_feature.square().mean().clamp_min(1e-6)
                loss = loss + response_weight * F.mse_loss(student_feature, teacher_feature) / scale
        if not math.isfinite(float(loss.detach())):
            raise FloatingPointError("Student distillation loss became nonfinite")
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 5, error_if_nonfinite=True)
        student_optimizer.step()
        total += float(loss.detach())
    return total / student_steps, float(generator_loss.detach()), synthetic



@torch.no_grad()
def synthetic_health(teacher, student, images, temperature=1):
    """Record collapse indicators without contributing to any update."""
    student_logits, features = cnn_outputs(student, images, full_responses=True)
    teacher_logits = teacher(images)
    hidden = features[2]
    probabilities = teacher_logits.softmax(1)
    return {
        "pixels_min": float(images.min()), "pixels_max": float(images.max()),
        "pixels_std": float(images.std()),
        "pixels_outside_normalized_range": float((images.abs() > 1).float().mean()),
        "teacher_entropy": float(-(probabilities * teacher_logits.log_softmax(1)).sum(1).mean()),
        "teacher_class_histogram": torch.bincount(teacher_logits.argmax(1), minlength=10).tolist(),
        "student_class_histogram": torch.bincount(student_logits.argmax(1), minlength=10).tolist(),
        "synthetic_agreement": float((teacher_logits.argmax(1) == student_logits.argmax(1)).float().mean()),
        "student_hidden_zero_fraction": float((hidden == 0).float().mean()),
        "student_hidden_all_zero_sample_fraction": float((hidden.abs().sum(1) == 0).float().mean()),
        "student_logit_std_across_inputs": float(student_logits.std(0).mean()),
        "synthetic_kl": float(divergence(student_logits, teacher_logits, temperature)),
    }


def student_is_collapsed(health):
    return (health["student_hidden_all_zero_sample_fraction"] >= 0.99
            and health["student_logit_std_across_inputs"] < 1e-6)


def train_student(args, teacher, test_images, test_labels, attack):
    seed_everything(args.seed + 10000)
    teacher.eval().requires_grad_(False)
    teacher_hash = model_digest(teacher)
    student = make_model(args.device)
    initial_hash = model_digest(student)
    if initial_hash == teacher_hash:
        raise RuntimeError("Student initialization unexpectedly matches the teacher")
    generator = ImageGenerator(output_mode=args.generator_output).to(args.device)
    student_optimizer = torch.optim.Adam(student.parameters(), lr=args.student_lr)
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.generator_lr)
    student_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(student_optimizer, args.distill_cycles)
    generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(generator_optimizer, args.distill_cycles)
    first_cycle = 1
    checkpoint_path = args.output / "distillation_checkpoint.pt"
    collapsed_evaluations = 0
    last_health = None
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
        if saved["teacher_sha256"] != teacher_hash:
            raise RuntimeError("Resumed distillation requires the same teacher checkpoint")
        student.load_state_dict(saved["student"])
        generator.load_state_dict(saved["generator"])
        student_optimizer.load_state_dict(saved["student_optimizer"])
        generator_optimizer.load_state_dict(saved["generator_optimizer"])
        student_scheduler.load_state_dict(saved["student_scheduler"])
        generator_scheduler.load_state_dict(saved["generator_scheduler"])
        first_cycle = saved["cycle"] + 1
        collapsed_evaluations = saved.get("collapsed_evaluations", 0)
        last_health = saved.get("synthetic_health")
        torch.set_rng_state(saved["torch_rng"].cpu())
        torch.cuda.set_rng_state(saved["cuda_rng"].cpu())
    else:
        initial = evaluate(student, test_images, test_labels, attack)
        record(args, "student_initial", 0, initial, elapsed_seconds=0)
        write_json(args.output / "student_initial_evaluation.json", initial)
        torch.save(copy.deepcopy(student).cpu(), args.output / "student_initial_model.pt")
    started = time.monotonic()
    for cycle in range(first_cycle, args.distill_cycles + 1):
        noise = torch.randn(args.batch_size, 100, device=args.device)
        loss, generator_loss, synthetic = distillation_step(
            teacher, student, generator, student_optimizer, generator_optimizer,
            noise, args.student_steps, args.temperature, args.attention_weight,
            args.generator_objective, args.confidence_weight, args.balance_weight,
            args.activation_weight, args.response_weight, args.response_preactivation,
            args.recompute_synthetic)
        student_scheduler.step()
        generator_scheduler.step()
        if cycle % args.log_every == 0 or cycle == 1:
            write_json(args.output / "progress.json", {
                "phase": "distillation", "cycle": cycle, "total_cycles": args.distill_cycles,
                "student_loss": loss, "generator_loss": generator_loss,
                "elapsed_seconds": time.monotonic() - started,
            })
            print(f"distillation {cycle}/{args.distill_cycles} loss={loss:.5f} "
                  f"G={generator_loss:.5f} elapsed={time.monotonic() - started:.1f}s", flush=True)
        if cycle % args.eval_every == 0 or cycle == args.distill_cycles:
            health = synthetic_health(teacher, student, synthetic, args.temperature)
            last_health = health
            write_json(args.output / f"synthetic_health_{cycle:06d}.json", health)
            collapsed_evaluations = collapsed_evaluations + 1 if student_is_collapsed(health) else 0
            metrics = evaluate(student, test_images, test_labels, attack)
            record(args, "student", cycle, metrics, loss=loss, generator_loss=generator_loss,
                   learning_rate=student_optimizer.param_groups[0]["lr"],
                   elapsed_seconds=time.monotonic() - started)
            save_image(synthetic[:16].detach().cpu().add(1).div(2).clamp(0, 1),
                       args.output / f"synthetic_cycle_{cycle:06d}.png", nrow=4)
            torch.save({"student": student.state_dict(), "generator": generator.state_dict(),
                        "student_optimizer": student_optimizer.state_dict(),
                        "generator_optimizer": generator_optimizer.state_dict(),
                        "student_scheduler": student_scheduler.state_dict(),
                        "generator_scheduler": generator_scheduler.state_dict(),
                        "teacher_sha256": teacher_hash, "cycle": cycle,
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state(),
                        "metrics": metrics, "synthetic_health": health,
                        "collapsed_evaluations": collapsed_evaluations}, checkpoint_path)
            if args.collapse_patience > 0 and collapsed_evaluations >= args.collapse_patience:
                write_json(args.output / "training_failure.json", {
                    "reason": "student_hidden_layer_collapsed", "cycle": cycle,
                    "consecutive_collapsed_evaluations": collapsed_evaluations,
                    "synthetic_health": health, "metrics": metrics,
                    "backdoor_conclusion": "invalid: normal task learning has collapsed",
                })
                raise RuntimeError("Student hidden layer remained zero with constant logits; "
                                   "checkpoint preserved. Inspect training_failure.json")
    if model_digest(teacher) != teacher_hash:
        raise RuntimeError("Teacher state changed during distillation")
    final = evaluate(student, test_images, test_labels, attack)
    write_json(args.output / "student_evaluation.json", final)
    write_json(args.output / "datafree_audit.json", {
        "teacher_sha256_before": teacher_hash, "teacher_sha256_after": model_digest(teacher),
        "student_initial_sha256": initial_hash, "student_final_sha256": model_digest(student),
        "student_training_inputs": "ImageGenerator(Gaussian noise) only",
        "student_training_real_samples": 0, "student_training_labeled_samples": 0,
        "test_split_usage": "no-grad evaluation at fixed intervals",
        "backdoor_preset": attack.spec.preset,
        "backdoor_spec": asdict(attack.spec),
        "generator_uses_trigger": False, "generator_uses_training_data": False,
        "distillation_preset": args.distillation_preset,
        "recompute_synthetic_each_student_step": args.recompute_synthetic,
        "generator_output": args.generator_output,
        "generator_objective": args.generator_objective,
        "attention_weight": args.attention_weight,
        "response_weight": args.response_weight,
        "response_preactivation": args.response_preactivation,
        "collapse_patience": args.collapse_patience,
        "final_synthetic_health": last_health,
        "student_steps": args.student_steps, "generator_steps_per_cycle": 1,
    })
    torch.save(copy.deepcopy(student).cpu(), args.output / "student_model.pt")
    torch.save(generator.state_dict(), args.output / "generator_state.pt")
    return final


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--distillation-preset", choices=("custom", "abd-zskt"), default="custom",
                        help="ABD ZSKT baseline defaults, with both ABD defenses disabled")
    parser.add_argument("--recompute-synthetic", action="store_true",
                        help="Recompute G on the reused noise for each student update, as in ABD")
    parser.add_argument("--phase", choices=("teacher", "distill", "full"), default="full")
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--teacher-epochs", type=int, default=80)
    parser.add_argument("--teacher-lr", type=float, default=0.05)
    parser.add_argument("--backdoor-preset", choices=("reference", "legacy"), default="reference")
    parser.add_argument("--attack-seed", type=int, default=0)
    parser.add_argument("--poison-ratio", type=float, default=None)
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--patch-location", default=None)
    parser.add_argument("--patch-margin", type=int, default=None)
    parser.add_argument("--distill-cycles", type=int, default=10000)
    parser.add_argument("--student-steps", type=int, default=10)
    parser.add_argument("--student-lr", type=float, default=0.002)
    parser.add_argument("--generator-lr", type=float, default=0.001)
    parser.add_argument("--generator-objective", choices=("adversarial", "dafl"), default="adversarial")
    parser.add_argument("--generator-output", choices=("unbounded", "tanh"), default="unbounded")
    parser.add_argument("--confidence-weight", type=float, default=1.0)
    parser.add_argument("--balance-weight", type=float, default=5.0)
    parser.add_argument("--activation-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--attention-weight", type=float, default=250)
    parser.add_argument("--response-weight", type=float, default=0,
                        help="Same-CNN control matching convolutional, hidden and logit responses")
    parser.add_argument("--response-preactivation", action="store_true",
                        help="Match linear responses before ReLU in the same-CNN control")
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--collapse-patience", type=int, default=3,
                        help="Stop after this many collapsed synthetic evaluations; 0 disables")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    preliminary, _ = parser.parse_known_args()
    if preliminary.distillation_preset == "abd-zskt":
        parser.set_defaults(distill_cycles=80000, student_steps=10, batch_size=128,
                            student_lr=0.002, generator_lr=0.001, temperature=1,
                            attention_weight=250, response_weight=0,
                            generator_output="unbounded", generator_objective="adversarial",
                            recompute_synthetic=True)
    args = parser.parse_args()
    reference_attack = args.backdoor_preset == 'reference'
    if args.poison_ratio is None:
        args.poison_ratio = 0.2 if reference_attack else 0.1
    if args.patch_size is None:
        args.patch_size = 4 if reference_attack else 5
    if args.patch_location is None:
        args.patch_location = 'lower_right' if reference_attack else 'upper_left'
    if args.patch_margin is None:
        args.patch_margin = 1 if reference_attack else 0
    if args.resume and (args.output / 'config.json').exists():
        old = json.loads((args.output / 'config.json').read_text())
        if old.get('backdoor_preset', 'legacy') != args.backdoor_preset:
            parser.error('Resume requires the same BadNets preset; historical runs use --backdoor-preset legacy')
    if args.phase == 'distill' and (args.output / 'teacher_data.json').exists():
        teacher_data = json.loads((args.output / 'teacher_data.json').read_text())
        if teacher_data.get('backdoor_preset', 'legacy') != args.backdoor_preset:
            parser.error('Teacher checkpoint uses a different BadNets preset; use its original preset')
    if args.collapse_patience < 0:
        parser.error("--collapse-patience must be nonnegative")
    if args.device != "cuda" or not torch.cuda.is_available():
        parser.error("This full-data experiment requires the available CUDA environment")
    args.output.mkdir(parents=True, exist_ok=True)
    config_path = args.output / "config.json"
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    if config_path.exists() and not args.resume:
        parser.error("Output directory already has an experiment. Use --resume or a new directory")
    if not config_path.exists():
        write_json(config_path, {**config, "architecture": "FedAvgCNN(3, 10, 1600)",
                                "torch_version": torch.__version__,
                                "gpu": torch.cuda.get_device_name(),
                                "distillation_reference": ("https://github.com/illidanlab/ABD/tree/master/zskt"
                                     if args.distillation_preset == "abd-zskt" else
                                     "https://github.com/polo5/ZeroShotKnowledgeTransfer")})
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)
    attack = make_attack(args)
    print(json.dumps(config, ensure_ascii=False), flush=True)
    test_images, test_labels = load_cifar10(args.data_root, False, args.device)
    if args.phase in ("teacher", "full"):
        teacher = train_teacher(args, test_images, test_labels, attack)
    else:
        teacher = make_model(args.device)
        saved = torch.load(args.output / "teacher_checkpoint.pt", map_location=args.device, weights_only=False)
        teacher.load_state_dict(saved["model"])
    if args.phase in ("distill", "full"):
        final = train_student(args, teacher, test_images, test_labels, attack)
        write_json(args.output / "completed.json", {"phase": "complete", "student": final})
    else:
        write_json(args.output / "teacher_completed.json", {"phase": "teacher_complete"})


if __name__ == "__main__":
    main()
