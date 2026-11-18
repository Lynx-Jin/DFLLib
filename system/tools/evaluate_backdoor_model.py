"""Evaluate clean accuracy and triggered ASR for one or more saved models."""

from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from torch.utils.data import DataLoader

from flcore.security.backdoor import BackdoorEvaluator, build_backdoor_attack
from utils.data_utils import read_client_data


class _ClientView:
    def __init__(self, data, batch_size):
        self._data = DataLoader(data, batch_size=batch_size, shuffle=False)

    def load_test_data(self):
        return self._data


def _load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser(description="Evaluate a saved backdoor model")
    parser.add_argument("--model", action="append", required=True,
                        help="saved torch.nn.Module; repeat for before/after")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--clients", type=int, required=True)
    parser.add_argument("--backdoor", default="badnets")
    parser.add_argument("--backdoor-base", default="badnets")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trigger-state", help="A3FL *_backdoor.pt saved alongside the model")
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--source-label", type=int, default=-1)
    parser.add_argument("--patch-size", type=int, default=5)
    parser.add_argument("--patch-location", default="upper_left")
    parser.add_argument("--blend-alpha", type=float, default=0.2)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    if args.clients <= 0 or args.batch_size <= 0:
        parser.error("clients and batch-size must be positive")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(args.device)
    attack_settings = SimpleNamespace(
        backdoor=args.backdoor,
        backdoor_base=args.backdoor_base,
        backdoor_target_label=args.target_label,
        backdoor_source_label=args.source_label,
        backdoor_patch_size=args.patch_size,
        backdoor_patch_location=args.patch_location,
        backdoor_blend_alpha=args.blend_alpha,
        backdoor_poison_ratio=0.0,
        backdoor_start_round=0,
        backdoor_end_round=-1,
        backdoor_dba_fragments=4,
        backdoor_seed=args.seed,
        backdoor_neurotoxin_mask_ratio=0.05,
        backdoor_f3ba_flip_ratio=0.01,
        backdoor_a3fl_step_size=0.01,
        backdoor_a3fl_shadow_steps=1,
    )
    trigger_patterns = None
    if args.trigger_state:
        state = _load(args.trigger_state, torch.device("cpu"))
        spec = state["spec"]
        trigger_patterns = state["patch_patterns"]
        if spec["name"] != "a3fl" or not trigger_patterns:
            raise ValueError("trigger-state must contain learned A3FL patches")
        settings = {"backdoor": spec["name"], "backdoor_base": spec["base_name"]}
        settings.update({f"backdoor_{key}": value for key, value in spec.items()
                         if key not in {"name", "base_name"}})
        attack_settings = SimpleNamespace(**settings)
    elif args.backdoor == "a3fl":
        parser.error("A3FL evaluation requires --trigger-state for its learned patches")
    eval_attack = build_backdoor_attack(
        attack_settings, client_id=None, is_malicious=False, evaluation=True)
    clients = [_ClientView(read_client_data(args.dataset, i, is_train=False, few_shot=0),
                           args.batch_size)
               for i in range(args.clients)]
    evaluator = BackdoorEvaluator(eval_attack, device)
    for path in args.model:
        model = _load(path, device)
        if not isinstance(model, torch.nn.Module):
            raise TypeError(f"{path} does not contain a serialized torch.nn.Module")
        model.to(device)
        eligible, clean_correct, triggered_target = evaluator.evaluate(
            model, clients, trigger_patterns=trigger_patterns)
        total, correct = 0, 0
        model.eval()
        with torch.no_grad():
            for client in clients:
                for x, y in client.load_test_data():
                    predictions = model(x.to(device)).argmax(dim=1).cpu()
                    correct += int((predictions == y).sum().item())
                    total += int(y.numel())
        clean_acc = correct / total if total else 0.0
        source_acc = clean_correct / eligible if eligible else 0.0
        asr = triggered_target / eligible if eligible else 0.0
        print(f"model={path}")
        print(f"samples={total} eligible={eligible} clean_accuracy={clean_acc:.6f} "
              f"source_accuracy={source_acc:.6f} asr={asr:.6f}")


if __name__ == "__main__":
    main()
