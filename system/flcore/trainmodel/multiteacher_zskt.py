"""Probability-ensemble ZSKT for the framework's identical CIFAR-10 CNNs."""
import copy
import math

import torch
from torch import nn
from torch.nn import functional as F

from tools.run_datafree_badnets import cnn_outputs, model_digest


def attention_map(feature):
    return F.normalize(feature.square().mean(1).flatten(1), dim=1)


class MultiTeacherEnsemble(nn.Module):
    """Freeze uploaded snapshots; average probabilities, not model parameters."""

    def __init__(self, models, weights):
        super().__init__()
        if not models or len(models) != len(weights):
            raise ValueError("An ensemble needs one weight per teacher")
        values = torch.as_tensor(weights, dtype=torch.float32,
                                 device=next(models[0].parameters()).device)
        if not bool(torch.isfinite(values).all()) or not bool((values > 0).all()):
            raise ValueError("Teacher weights must be finite and positive")
        self.models = nn.ModuleList(copy.deepcopy(model).eval().requires_grad_(False)
                                    for model in models)
        self.register_buffer("weights", values / values.sum())
        self.eval()

    def train(self, mode=True):
        # Frozen teacher BN/dropout state must never change during G training.
        return super().train(False)

    def distribution(self, images, temperature=1.0, with_attention=False):
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("Temperature must be positive")
        log_probabilities = []
        maps = None
        for weight, model in zip(self.weights, self.models):
            if with_attention:
                logits, features = cnn_outputs(model, images)
                current = [attention_map(feature) for feature in features]
                if maps is None:
                    maps = [weight * value for value in current]
                else:
                    maps = [total + weight * value for total, value in zip(maps, current)]
            else:
                logits = model(images)
            log_probabilities.append(F.log_softmax(logits.float() / temperature, dim=1)
                                     + weight.log())
        return torch.logsumexp(torch.stack(log_probabilities), dim=0), maps

    def forward(self, images):
        return self.distribution(images)[0]

    def digests(self):
        return [model_digest(model) for model in self.models]


def ensemble_kl(student_logits, teacher_log_probabilities, temperature=1.0):
    # ZSKT's batch-and-class mean KL, with stable differentiable log targets.
    return F.kl_div(F.log_softmax(student_logits.float() / temperature, dim=1),
                    teacher_log_probabilities, log_target=True,
                    reduction="batchmean") / student_logits.shape[1]


def multiteacher_step(ensemble, student, generator, student_optimizer,
                      generator_optimizer, noise, student_steps=10,
                      temperature=1.0, attention_weight=250.0):
    """All update inputs are G(noise); no dataset, class labels or trigger."""
    ensemble.eval()
    student.train().requires_grad_(False)
    generator.train()
    generator_optimizer.zero_grad(set_to_none=True)
    synthetic = generator(noise)
    target, _ = ensemble.distribution(synthetic, temperature)
    generator_loss = -ensemble_kl(student(synthetic), target, temperature)
    if not bool(torch.isfinite(generator_loss)):
        raise FloatingPointError("Multi-teacher generator loss became nonfinite")
    generator_loss.backward()
    nn.utils.clip_grad_norm_(generator.parameters(), 5, error_if_nonfinite=True)
    generator_optimizer.step()
    student.requires_grad_(True)
    with torch.no_grad():
        synthetic = generator(noise).detach()
        target, teacher_maps = ensemble.distribution(synthetic, temperature,
                                                     with_attention=bool(attention_weight))
    total = 0.0
    for _ in range(student_steps):
        student_optimizer.zero_grad(set_to_none=True)
        logits, features = cnn_outputs(student, synthetic)
        loss = ensemble_kl(logits, target, temperature)
        if attention_weight:
            loss = loss + attention_weight * sum(
                F.mse_loss(attention_map(feature), teacher_map)
                for feature, teacher_map in zip(features, teacher_maps))
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Multi-teacher student loss became nonfinite")
        loss.backward()
        nn.utils.clip_grad_norm_(student.parameters(), 5, error_if_nonfinite=True)
        student_optimizer.step()
        total += float(loss.detach())
    return total / student_steps, float(generator_loss.detach()), synthetic
