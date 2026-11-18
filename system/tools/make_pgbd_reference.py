"""Create a small clean labelled reference set for the PGBD post-hoc test.

This helper is intended for simulator experiments only.  In a real
deployment the server should provide its own clean reference file rather than
pooling client-held-out partitions.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import torch

from utils.data_utils import read_client_data


def main():
    parser = argparse.ArgumentParser(description="Build a PGBD clean reference npz")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--samples-per-class", type=int, default=10)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.clients <= 0 or args.classes <= 1 or args.samples_per_class <= 0:
        raise ValueError("clients/classes/samples-per-class must be positive")

    buckets = [[] for _ in range(args.classes)]
    for client_id in range(args.clients):
        for x, y in read_client_data(args.dataset, client_id, is_train=False, few_shot=0):
            label = int(y.item())
            if 0 <= label < args.classes and len(buckets[label]) < args.samples_per_class:
                buckets[label].append(x.detach().cpu())

    missing = [label for label, values in enumerate(buckets) if not values]
    if missing:
        raise ValueError(f"no reference examples found for classes {missing}")
    x = torch.stack([value for values in buckets for value in values]).numpy()
    y = torch.tensor([
        label for label, values in enumerate(buckets) for _ in values
    ], dtype=torch.int64).numpy()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    np.savez_compressed(args.output, x=x, y=y)
    print(f"saved={args.output} samples={len(y)} classes={args.classes}")


if __name__ == "__main__":
    main()
