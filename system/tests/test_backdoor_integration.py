"""Regressions for the shared image-backdoor training and evaluation paths."""

import copy
from contextlib import contextmanager, redirect_stdout
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.data import DataLoader

from flcore.security.backdoor import BackdoorDataset, BackdoorEvaluator, build_backdoor_attack
from flcore.security.monitor import UpdateMonitor
from flcore.servers.serveravg import FedAvg
from flcore.servers.serverfedsgd import FedSGD
from test_port_integration import parse_cli


IMAGES = [
    (torch.full((1, 4, 4), -0.5), torch.tensor(0)),
    (torch.full((1, 4, 4), 0.5), torch.tensor(1)),
] * 2


@contextmanager
def image_server(backdoor="badnets", algorithm="FedAvg", extra=()):
    args = parse_cli([
        "-algo", algorithm, "-dev", "cpu", "-nc", "4", "-jr", "1",
        "-ncl", "2", "-lbs", "4", "-amr", "0.5", "-bd", backdoor,
        "--backdoor_patch_size", "2", "--backdoor_poison_ratio", "1", *extra,
    ])
    torch.manual_seed(29)
    args.model = nn.Sequential(nn.Flatten(), nn.Linear(16, 2))
    with tempfile.TemporaryDirectory() as folder, redirect_stdout(io.StringIO()):
        args.experiment_log_dir = folder
        args.save_folder_name = str(Path(folder) / "items")
        args.attack_monitor_dir = str(Path(folder) / "monitor")
        with patch("flcore.servers.serverbase.read_client_data", return_value=IMAGES), \
                patch("flcore.clients.clientbase.read_client_data", return_value=IMAGES), \
                patch.object(UpdateMonitor, "plot"):
            server_class = FedSGD if algorithm == "FedSGD" else FedAvg
            server = server_class(args, 0)
            server.selected_clients = server.clients
            yield server


class BackdoorIntegrationTest(unittest.TestCase):
    def test_all_presets_train_through_both_upload_protocols(self):
        for algorithm in ("FedAvg", "FedSGD"):
            for name in ("badnets", "blended", "refool", "clean_label", "dba",
                         "neurotoxin", "a3fl", "f3ba"):
                with self.subTest(algorithm=algorithm, backdoor=name), \
                        image_server(name, algorithm) as server:
                    for round_id in range(2):
                        server.send_models()
                        for client in server.clients:
                            self.assertEqual(client.backdoor.round_id, round_id)
                            client.train()
                        server.receive_models()
                        server.aggregate_parameters()
                        self.assertTrue(all(torch.isfinite(value).all()
                                            for value in server.global_model.parameters()))
                    self.assertEqual(len(server.malicious_ids), 2)
                    self.assertTrue((Path(server.update_monitor.output_dir) /
                                     "round_audit.csv").is_file())
                    if name == "a3fl":
                        self.assertTrue(all(server.clients[cid].backdoor.patch_pattern is not None
                                            for cid in server.malicious_ids))

    def test_schedule_advances_even_without_monitor_and_test_data_stays_clean(self):
        with image_server(extra=("--backdoor_start_round", "1",
                                 "--backdoor_end_round", "1")) as server:
            server.update_monitor = None
            malicious = server.clients[server.malicious_ids[0]]
            benign = next(client for client in server.clients if not client.is_malicious)
            for round_id in range(3):
                server.send_models()
                self.assertEqual(isinstance(malicious.load_train_data().dataset, BackdoorDataset),
                                 round_id == 1)
                self.assertFalse(isinstance(benign.load_train_data().dataset, BackdoorDataset))
                self.assertFalse(isinstance(malicious.load_test_data().dataset, BackdoorDataset))
                for client in server.clients:
                    client.train()
                server.receive_models()
                server.aggregate_parameters()
            self.assertEqual(server.update_monitor_round, 3)

    def test_poison_ratio_zero_and_clean_label_policy(self):
        for name in ("badnets", "refool", "clean_label"):
            with self.subTest(name=name), image_server(name) as server:
                server.send_models()
                client = server.clients[server.malicious_ids[0]]
                dataset = client.load_train_data().dataset
                for index in dataset.poison_indices:
                    image, label = dataset[index]
                    if name == "badnets":
                        self.assertEqual(int(label), 0)
                    else:
                        self.assertEqual(int(label), int(IMAGES[index][1]))
                server.args.backdoor_poison_ratio = 0
                attack = build_backdoor_attack(server.args, client.id, True)
                self.assertEqual(attack.choose_poison_indices(IMAGES, 0), set())

    def test_dba_coalition_union_matches_full_patch(self):
        with image_server("dba") as server:
            original = torch.zeros((1, 4, 4))
            fragments = []
            for client_id in server.malicious_ids:
                fragments.append(server.clients[client_id].backdoor.apply_trigger(original))
            # Four fragments require four colluders to reconstruct the full patch.
            attacks = []
            for client_id in range(4):
                attack = build_backdoor_attack(server.args, client_id, True)
                attack.configure_coalition(range(4))
                attacks.append(attack.apply_trigger(original))
            union = torch.stack(attacks).amax(dim=0)
            expected = server.backdoor_evaluator.attack.apply_trigger(original, union=True)
            torch.testing.assert_close(union, expected)
            self.assertEqual(len(fragments), 2)

    def test_neurotoxin_uses_previous_global_coordinates(self):
        with image_server("neurotoxin", extra=("--backdoor_neurotoxin_mask_ratio", "0.5")) as server:
            client = server.clients[server.malicious_ids[0]]
            model = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                model.weight.copy_(torch.tensor([[3.0, 0.1]]))
            client.backdoor.begin_round(1, {"weight": torch.zeros_like(model.weight)})
            model.weight.grad = torch.ones_like(model.weight)
            client.backdoor.mask_gradients(model)
            torch.testing.assert_close(model.weight.grad, torch.tensor([[0.0, 1.0]]))

    def test_f3ba_hook_is_disabled_on_benign_clients(self):
        with image_server("f3ba", extra=("--backdoor_f3ba_flip_ratio", "0.5")) as server:
            malicious = server.clients[server.malicious_ids[0]].backdoor
            benign = next(client.backdoor for client in server.clients if not client.is_malicious)
            model = nn.Linear(2, 1, bias=False)
            with torch.no_grad():
                model.weight.copy_(torch.tensor([[0.1, 2.0]]))
            other = copy.deepcopy(model)
            malicious.before_local_train(model)
            benign.before_local_train(other)
            torch.testing.assert_close(model.weight, torch.tensor([[-0.1, 2.0]]))
            torch.testing.assert_close(other.weight, torch.tensor([[0.1, 2.0]]))

    def test_evaluator_excludes_target_class_and_preserves_model_mode(self):
        class PatchClassifier(nn.Module):
            def forward(self, images):
                patched = images[:, 0, 0, 0]
                return torch.stack((patched - 0.75, 0.75 - patched), dim=1)

        with image_server(extra=("--backdoor_patch_location", "upper_left",
                                 "--backdoor_patch_margin", "0")) as server:
            model = PatchClassifier().train()
            evaluator = BackdoorEvaluator(server.backdoor_evaluator.attack, torch.device("cpu"))
            client = server.clients[0]
            client.load_test_data = lambda: DataLoader(IMAGES, batch_size=4)
            eligible, correct, triggered = evaluator.evaluate(model, [client])
            self.assertEqual((eligible, correct, triggered), (2, 2, 2))
            self.assertTrue(model.training)

    def test_a3fl_evaluation_uses_learned_patterns_and_saves_them(self):
        class PatchClassifier(nn.Module):
            def forward(self, images):
                values = images[:, 0, 0, 0]
                return torch.stack((values - 0.75, 0.75 - values), dim=1)

        with image_server("a3fl") as server:
            for index, client_id in enumerate(server.malicious_ids):
                server.clients[client_id].backdoor.patch_pattern = torch.full((1, 4, 4), float(index))
            patterns = server._backdoor_trigger_patterns()
            evaluator = server.backdoor_evaluator
            counts = evaluator.evaluate(PatchClassifier(), [server.clients[0]], trigger_patterns=patterns)
            self.assertEqual(counts, (4, 4, 2))
            self.assertIsNone(evaluator.attack.patch_pattern)
            server.save_global_model()
            state_file = Path(server._global_model_path()[:-3] + "_backdoor.pt")
            state = torch.load(state_file, weights_only=True)
            self.assertEqual(state["spec"]["name"], "a3fl")
            self.assertEqual(len(state["patch_patterns"]), 2)
            torch.testing.assert_close(state["patch_patterns"], patterns)

    def test_invalid_backdoor_configuration_fails_before_training(self):
        for extra in (("--backdoor_poison_ratio", "-0.1"),
                      ("--backdoor_dba_fragments", "0"),
                      ("--backdoor_start_round", "3", "--backdoor_end_round", "1"),
                      ("--backdoor_target_label", "10"),
                      ("-atk", "poisonedfl"), ("-algo", "FedGen")):
            with self.subTest(flags=extra), self.assertRaises(ValueError):
                parse_cli(["-algo", "FedAvg", "-bd", "badnets", *extra])


if __name__ == "__main__":
    unittest.main()
