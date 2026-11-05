"""Scalar cosine oracle for the fixed-direction NDSS 2021 attack framework.

The proxy-quantile oracle is an adaptation to partial knowledge of this
project's detector, not the paper's AFA selection oracle. No KMeans output,
honest-client updates, or AGR-agnostic Min-Max distance bound is used here.
"""

from collections import namedtuple
import math

import numpy as np
import torch


CosineSearchResult = namedtuple(
    "CosineSearchResult", "lambda_value cosine target_reached cap_reached")


def _real_roots(coefficients):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    scale = float(np.max(np.abs(coefficients)))
    if scale == 0.0:
        return []
    coefficients = coefficients / scale
    # Remove only exact leading zeros: a small quadratic term may still
    # determine a boundary inside a sufficiently large search interval.
    coefficients = np.trim_zeros(coefficients, trim="f")
    return [
        float(root.real) for root in np.roots(coefficients)
        if abs(root.imag) <= 1e-10 * max(1.0, abs(root.real))
    ]


def strongest_cosine_point(global_vector, mean, deviation, target, lambda_cap):
    """Maximize lambda in [0, cap] with cos(W + mean - lambda*d, W) >= target.

    Squaring the cosine boundary gives a quadratic. Partition at every real
    root, then check the *signed* cosine on each interval. This also handles
    feasible intervals that start after lambda=0, without a monotonicity
    assumption or a grid whose resolution depends on the deviation type.
    """
    reference = global_vector.detach().to(dtype=torch.float64)
    base = reference + mean.detach().to(dtype=torch.float64)
    direction = -deviation.detach().to(dtype=torch.float64)
    reference_norm = float(torch.norm(reference).item())
    if reference_norm <= 1e-15:
        raise ValueError("A nonzero global model is required for a cosine oracle")

    projection_base = float(torch.dot(base, reference).item()) / reference_norm
    projection_dir = float(torch.dot(direction, reference).item()) / reference_norm
    base_sq = float(torch.dot(base, base).item())
    cross = float(torch.dot(base, direction).item())
    direction_sq = float(torch.dot(direction, direction).item())

    def cosine_at(value):
        norm_sq = base_sq + 2.0 * value * cross + value * value * direction_sq
        if norm_sq <= 1e-24:
            return -math.inf  # cosine of a zero vector is undefined
        value = (projection_base + value * projection_dir) / math.sqrt(norm_sq)
        return max(-1.0, min(1.0, value))

    boundaries = [0.0, float(lambda_cap)]
    target_sq = float(target) ** 2
    roots = _real_roots([
        projection_dir ** 2 - target_sq * direction_sq,
        2.0 * (projection_base * projection_dir - target_sq * cross),
        projection_base ** 2 - target_sq * base_sq,
    ])
    # Signed numerator and zero-model boundaries remove roots introduced by
    # squaring and prevent accepting a direction with the wrong sign.
    roots += _real_roots([projection_dir, projection_base])
    roots += _real_roots([direction_sq, 2.0 * cross, base_sq])
    boundaries += [value for value in roots if 0.0 < value < lambda_cap]
    boundaries = sorted(set(boundaries))

    feasible = [value for value in boundaries if cosine_at(value) >= target - 1e-12]
    for left, right in zip(boundaries, boundaries[1:]):
        if cosine_at((left + right) / 2.0) >= target:
            # Stay a little inside the feasible interval to absorb float32
            # rounding when the actual uploaded model is reconstructed.
            endpoint = right
            if cosine_at(right) < target or right < lambda_cap:
                endpoint = right - (right - left) * 1e-7
            if cosine_at(endpoint) >= target - 1e-12:
                feasible.append(endpoint)

    if feasible:
        best = max(feasible)
        return CosineSearchResult(
            best, cosine_at(best), cosine_at(best) >= target - 1e-12,
            best >= lambda_cap * (1.0 - 1e-10))

    # The requested target can be unreachable on this attack ray. Return the
    # highest-cosine candidate and explicitly report failure, rather than
    # silently labelling a clean-mean fallback as a successful attack.
    candidates = [0.0, float(lambda_cap)]
    denominator = projection_dir * cross - projection_base * direction_sq
    if denominator != 0.0:
        stationary = (projection_base * cross - projection_dir * base_sq) / denominator
        if 0.0 < stationary < lambda_cap:
            candidates.append(stationary)
    best = max(candidates, key=lambda value: (cosine_at(value), value))
    return CosineSearchResult(best, cosine_at(best), False, best == lambda_cap)
