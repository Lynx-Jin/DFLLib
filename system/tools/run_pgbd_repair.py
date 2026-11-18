"""Run PGBD post-hoc repair on a saved PFLlib model.

Example (from ``system/``)::

    python -m tools.run_pgbd_repair \
        --model ../results/.../models/MNIST_100_FedAvg_server_run0.pt \
        --reference-npz ../dataset/MNIST_100/pgbd_clean_reference.npz \
        --classes 10 --layer fc1 --target-label 0 --device cuda \
        --output ../results/.../models/pgbd_repaired.pt

The input model is expected to be a serialized ``torch.nn.Module`` (the
format emitted by ``Server.save_global_model``).  The reference npz can use
either ``x``/``y`` arrays or PFLlib's ``data`` object with ``x``/``y`` fields.
"""

from __future__ import annotations

import argparse
import os
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from flcore.security.defense.pgbd_repair import PGBDRepair, load_reference_npz
from flcore.security.backdoor import build_backdoor_attack


def _load_model(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # PyTorch versions before the weights_only argument.
        return torch.load(path, map_location=device)


@torch.no_grad()
def _accuracy(model, reference_data, device, batch_size):
    model.eval()
    loader = DataLoader(reference_data, batch_size=batch_size, shuffle=False)
    total, correct = 0, 0
    for x, y in loader:
        logits = model(x.to(device))
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        correct += int((logits.argmax(dim=1).cpu() == y).sum().item())
        total += int(y.numel())
    return float(correct / total) if total else 0.0


def main():
    parser = argparse.ArgumentParser(description="PGBD post-hoc model repair")
    parser.add_argument("--model", required=True, help="serialized poisoned nn.Module")
    parser.add_argument("--reference-npz", required=True, help="clean x/y reference npz")
    parser.add_argument("--triggered-reference-npz",
                        help="Optional precomputed labeled references for ST-PGBD prototypes")
    parser.add_argument("--output", required=True, help="path for repaired model")
    parser.add_argument("--classes", type=int, required=True)
    parser.add_argument("--layer", default="", help="activation layer, e.g. fc1")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--pav-weight", type=float, default=1.0)
    parser.add_argument("--clean-loss-weight", type=float, default=1.0)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--trigger-backdoor", default="none",
                        choices=["none", "badnets", "blended", "refool", "dba"],
                        help="known trigger used to build ST-PGBD triggered prototypes")
    parser.add_argument("--trigger-patch-size", type=int, default=5)
    parser.add_argument("--trigger-patch-location", default="upper_left")
    parser.add_argument("--trigger-blend-alpha", type=float, default=0.2)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.triggered_reference_npz and args.trigger_backdoor != "none":
        parser.error("choose one source for the triggered reference")

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for PGBD but this Python environment has no CUDA")
    device = torch.device(args.device)
    reference = load_reference_npz(args.reference_npz)
    poisoned_reference = (load_reference_npz(args.triggered_reference_npz)
                          if args.triggered_reference_npz else None)
    if args.trigger_backdoor != "none":
        trigger = build_backdoor_attack(SimpleNamespace(
            backdoor=args.trigger_backdoor,
            backdoor_base=args.trigger_backdoor,
            backdoor_target_label=args.target_label,
            backdoor_source_label=-1,
            backdoor_poison_ratio=1.0,
            backdoor_start_round=0,
            backdoor_end_round=-1,
            backdoor_patch_size=args.trigger_patch_size,
            backdoor_patch_location=args.trigger_patch_location,
            backdoor_blend_alpha=args.trigger_blend_alpha,
            backdoor_dba_fragments=4,
            backdoor_seed=0,
            backdoor_neurotoxin_mask_ratio=0.05,
            backdoor_f3ba_flip_ratio=0.01,
            backdoor_a3fl_step_size=0.01,
            backdoor_a3fl_shadow_steps=1,
        ), client_id=None, is_malicious=False, evaluation=True)
        poisoned_reference = [(trigger.apply_trigger(x, union=True), y)
                              for x, y in reference]
    model = _load_model(args.model, device)
    if not isinstance(model, torch.nn.Module):
        raise TypeError("--model must contain a serialized torch.nn.Module")
    model.to(device)

    before = _accuracy(model, reference, device, args.batch_size)
    repairer = PGBDRepair(
        num_classes=args.classes,
        device=device,
        layer_name=args.layer or None,
        target_label=args.target_label,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        pav_weight=args.pav_weight,
        clean_loss_weight=args.clean_loss_weight,
        weight_decay=args.weight_decay,
    )
    result = repairer.repair(
        model, reference, poisoned_reference_data=poisoned_reference)
    after = _accuracy(model, reference, device, args.batch_size)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    torch.save(model, args.output)
    print(f"PGBD layer={result.layer_name} target={result.target_label}")
    print(f"reference_samples={result.num_reference_samples}")
    print(f"clean_reference_accuracy_before={before:.6f}")
    print(f"clean_reference_accuracy_after={after:.6f}")
    print("epoch_total_loss=" + ",".join(f"{value:.6g}" for value in result.losses))
    print("epoch_pav_loss=" + ",".join(f"{value:.6g}" for value in result.pav_losses))
    print(f"saved={args.output}")


if __name__ == "__main__":
    main()
