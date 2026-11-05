"""Exercise DFLlib's actual clients, receive path, monitor and optimizers."""
import argparse
import ast
import copy
from contextlib import contextmanager, redirect_stdout
import io
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch.utils.data import DataLoader

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverbase import Server
from flcore.servers.serverfedsgd import FedSGD
from flcore.security.attack.poisonedfl import flatten_model, write_model
from flcore.security.defense.scalar_model import ScalarModelDefense, submitted_model_scalars
from flcore.security.monitor import UpdateMonitor


DATA = [(torch.tensor([1., 0.]), torch.tensor(0)),
        (torch.tensor([0., 1.]), torch.tensor(1)),
        (torch.tensor([.4, .8]), torch.tensor(1))]


def parse_cli(argv):
    # Compile the actual parser and validator without importing all optional
    # PFLlib algorithm dependencies or executing training/log initialization.
    path = Path(__file__).resolve().parents[1] / 'main.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'))
    guard = next(n for n in tree.body if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Compare)
                 and isinstance(n.test.left, ast.Name) and n.test.left.id == '__name__')
    statements = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                  and n.name == 'validate_security_options']
    for n in guard.body:
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'parser' for t in n.targets):
            statements.append(n)
        elif (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
              and isinstance(n.value.func, ast.Attribute)
              and isinstance(n.value.func.value, ast.Name) and n.value.func.value.id == 'parser'):
            statements.append(n)
    ns = dict(argparse=argparse, math=math, DEFAULT_POISONEDFL_WINDOW=10)
    exec(compile(ast.Module(body=statements, type_ignores=[]), str(path), 'exec'), ns)
    args = ns['parser'].parse_args(argv)
    ns['validate_security_options'](args)
    return args


@contextmanager
def server_fixture(algorithm='FedSGD', optimizer='adam', attack='none', defense='none',
                   count=6, ratio=0., start=0, model=None):
    with tempfile.TemporaryDirectory() as tmp, redirect_stdout(io.StringIO()):
        flags = ['-algo', algorithm, '-fopt', optimizer, '-flr', '.001', '-data', 'MNIST',
                 '-nc', str(count), '-jr', '1', '-lbs', '2', '-atk', attack,
                 '-amr', str(ratio), '-def', defense, '-defstart', str(start),
                 '-scmin', '3', '-scobs', '1', '-schist', '2', '-scfreq', '1',
                 '-pfw', '2', '-dev', 'cpu']
        if attack == 'poisonedfl' and optimizer == 'adam' and algorithm == 'FedSGD':
            flags += ['-pfgm', 'adam_proxy']
        args = parse_cli(flags)
        torch.manual_seed(123)
        args.model = torch.nn.Linear(2, 2) if model is None else model
        args.num_classes = 2
        args.experiment_log_dir = tmp
        args.save_folder_name = str(Path(tmp)/'items')
        args.attack_monitor_dir = str(Path(tmp)/'monitor')
        with patch('flcore.servers.serverbase.read_client_data', return_value=DATA), \
                patch.object(UpdateMonitor, 'plot'):
            server = (FedSGD if algorithm == 'FedSGD' else FedAvg)(args, 0)
            server.selected_clients = server.clients
            yield server


class PortIntegrationTest(unittest.TestCase):
    def test_poisonedfl_fixed_population_and_all_three_upload_mappings(self):
        for algorithm, optimizer in [('FedAvg', 'sgd'), ('FedSGD', 'sgd'), ('FedSGD', 'adam')]:
            with self.subTest(algorithm=algorithm, optimizer=optimizer), server_fixture(
                    algorithm, optimizer, 'poisonedfl', count=20, ratio=.2) as server:
                self.assertEqual(server.num_clients, 20)
                self.assertEqual(len(server.clients), 20)
                self.assertEqual(server.malicious_ids, [1, 8, 18, 19])
                for client in server.clients:
                    if client.is_malicious:
                        client.load_train_data = Mock(side_effect=AssertionError('Attacker read training data'))
                    else:
                        client.load_train_data = lambda **kw: DataLoader(
                            DATA, batch_size=2, drop_last=kw.get('drop_last', True), shuffle=False)
                for _ in range(4):
                    server.send_models()
                    before = flatten_model(server.global_model).clone()
                    for client in server.clients:
                        client.train()
                    server.receive_models()
                    payloads = []
                    for cid, carrier in zip(server.uploaded_ids, server.uploaded_models):
                        if cid in server.malicious_ids:
                            vector = flatten_model(carrier)
                            payloads.append(vector)
                            delta = server.poisonedfl_attack.last_malicious_update
                            expected = (before + delta if algorithm == 'FedAvg' else
                                        before - delta if optimizer == 'adam' else before - delta/.001)
                            torch.testing.assert_close(vector, expected)
                    self.assertTrue(all(torch.equal(payloads[0], row) for row in payloads[1:]))
                    server.aggregate_parameters()
                    self.assertTrue(torch.isfinite(flatten_model(server.global_model)).all())
                records = [json.loads(line) for line in (Path(server.security_log_dir)/'attack_trace.jsonl').read_text().splitlines()]
                self.assertEqual([r['round'] for r in records], [0, 1, 2, 3])
                self.assertTrue(records[2]['attack']['feedback_checked'])
                self.assertEqual(records[-1]['attack']['model_delta_equivalent'], optimizer != 'adam')

    def test_native_model_scalars_do_not_use_delta_or_weighted_mean_reference(self):
        model = torch.nn.Linear(2, 1, bias=False)
        write_model(model, torch.tensor([1., 0.]))
        first, second = copy.deepcopy(model), copy.deepcopy(model)
        write_model(first, torch.tensor([1., 1.]))
        write_model(second, torch.tensor([2., 0.]))
        norms, cosines = submitted_model_scalars(model, [first, second], [10, 30])
        self.assertAlmostEqual(norms[10], math.sqrt(2), places=6)
        self.assertAlmostEqual(cosines[10], 1/math.sqrt(2), places=6)
        self.assertEqual(norms[30], 2.)
        self.assertEqual(cosines[30], 1.)

    def test_majority_mask_reweights_real_fedavg_and_ignores_truth_labels(self):
        model = torch.nn.Linear(2, 1, bias=False)
        write_model(model, torch.tensor([1., 0.]))
        with server_fixture('FedAvg', 'sgd', defense='scalar_kmeans', count=5, model=model) as server:
            outcomes = []
            for truth in [[], [0, 1, 2]]:
                server.malicious_ids = truth
                for cid, client in enumerate(server.clients):
                    client.train_samples = cid + 1
                    write_model(client.model, torch.tensor([1., [.1, .2, .3, 1., 1.][cid]]))
                server.receive_models()
                outcomes.append(set(server.uploaded_ids))
            self.assertEqual(outcomes, [{0, 1, 2}, {0, 1, 2}])
            weights = dict(zip(server.uploaded_ids, server.uploaded_weights))
            for cid in (0, 1, 2):
                self.assertAlmostEqual(weights[cid], (cid+1)/6)
            server.aggregate_parameters()
            torch.testing.assert_close(flatten_model(server.global_model), torch.tensor([1., 1.4/6]))

    def test_all_rejected_after_warmup_preserves_adam_model_and_state(self):
        model = torch.nn.Linear(2, 1, bias=False)
        write_model(model, torch.tensor([1., 0.]))
        with server_fixture(defense='scalar_dbscan', count=6, start=1, model=model) as server:
            for client in server.clients:
                write_model(client.model, torch.tensor([1., .2]))
            server.receive_models()  # warm-up: candidate groups observed, no rejection
            self.assertEqual(len(server.uploaded_ids), 6)
            self.assertTrue(server.robust_defense.last_details['confirmed_ids'])
            server.aggregate_parameters()
            before = copy.deepcopy(server.global_model.state_dict())
            steps = [v['step'].clone() for v in server.server_optimizer.state.values()]
            for client in server.clients:
                write_model(client.model, flatten_model(server.global_model)+torch.tensor([0., .2]))
            server.receive_models()
            self.assertEqual(server.uploaded_ids, [])
            self.assertTrue(server.skip_aggregation)
            server.aggregate_parameters()
            for key, value in before.items():
                self.assertTrue(torch.equal(value, server.global_model.state_dict()[key]))
            for expected, value in zip(steps, server.server_optimizer.state.values()):
                self.assertTrue(torch.equal(expected, value['step']))

    def test_scalar_adapter_uses_real_ids_when_upload_order_changes(self):
        with redirect_stdout(io.StringIO()):
            defense = ScalarModelDefense(min_group=3, min_history=2, history_window=3)
            ids = [8, 9, 10, 11, 12]
            norms = {cid: 1. for cid in ids}
            cosines = {8: .9, 9: .9, 10: .9, 11: .90001, 12: .90002}
            # All in one broad majority branch; exact groups retain ID histories.
            defense.filter_scalars(norms, cosines, ids, filtering_enabled=False)
            defense.filter_scalars(norms, cosines, list(reversed(ids)), filtering_enabled=False)
            counts = defense.last_details['history_observation_counts']
            self.assertEqual(counts[8], 2)
            self.assertEqual(counts[9], 2)
            self.assertEqual(counts[10], 2)
            with self.assertRaises(ValueError):
                defense.filter_scalars(norms, cosines, [8, 8, 10, 11, 12])

    def test_invalid_adam_mapping_and_wrong_attack_variant_rejected(self):
        alias = parse_cli(['-algo', 'FedSGD', '-fopt', 'adam', '-atk', 'Poisoned_FL',
                           '-pfgm', 'adam_proxy'])
        self.assertEqual(alias.attack_type, 'poisonedfl')
        with self.assertRaisesRegex(ValueError, 'adam_proxy'):
            parse_cli(['-algo', 'FedSGD', '-fopt', 'adam', '-atk', 'poisonedfl'])
        with self.assertRaisesRegex(ValueError, 'minmax'):
            parse_cli(['-atk', 'minsum', '-amv', 'cosine_oracle'])
        with self.assertRaisesRegex(ValueError, 'positive'):
            parse_cli(['-def', 'scalar_dbscan', '-schr', 'nan'])
        with server_fixture(attack='poisonedfl', ratio=.2) as server:
            server.args.attack_poisonedfl_gradient_mode = 'sgd_delta'
            with self.assertRaisesRegex(ValueError, 'adam_proxy'):
                Server(server.args, 1)


if __name__ == '__main__':
    unittest.main()
