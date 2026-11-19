import contextlib
import copy
import io
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from flcore.clients.clientavgzskt import AugmentedBackdoorDataset
from flcore.security.backdoor import build_backdoor_attack
from flcore.trainmodel.multiteacher_zskt import (
    MultiTeacherEnsemble, ensemble_kl, multiteacher_step, attention_map,
)
from tools.prepare_federated_zskt import balanced_iid_indices, prepare, prepare_legacy
from tools.run_datafree_badnets import (ImageGenerator, make_model, model_digest,
                                        cnn_outputs, divergence)


class ConstantLogits(nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, images):
        return self.values.expand(len(images), -1)


class FederatedZSKTTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(4)
        torch.manual_seed(12)

    def test_ensemble_is_a_probability_mixture_not_an_average_of_logits(self):
        models = [ConstantLogits([10, 0]), ConstantLogits([0, 0])]
        ensemble = MultiTeacherEnsemble(models, [1, 3])
        actual, _ = ensemble.distribution(torch.zeros(2, 3, 32, 32))
        expected = .25 * models[0].values.softmax(0) + .75 * models[1].values.softmax(0)
        torch.testing.assert_close(actual.exp(), expected.expand(2, -1))
        self.assertGreater(float((actual.exp()[0] - torch.tensor([2.5, 0]).softmax(0)).abs().max()), .1)
        ensemble.train()
        self.assertTrue(all(not teacher.training for teacher in ensemble.models))
        self.assertTrue(all(not parameter.requires_grad for parameter in ensemble.parameters()))
        self.assertTrue(all(parameter.requires_grad for teacher in models for parameter in teacher.parameters()))

    def test_one_teacher_reduces_to_single_teacher_kl_and_attention(self):
        teacher, student = make_model('cpu'), make_model('cpu')
        images = torch.randn(2, 3, 32, 32)
        ensemble = MultiTeacherEnsemble([teacher], [1])
        for temperature in [1, 3]:
            target, maps = ensemble.distribution(images, temperature, True)
            logits, features = cnn_outputs(teacher, images)
            torch.testing.assert_close(ensemble_kl(student(images), target, temperature),
                                       divergence(student(images), logits, temperature))
            for actual, feature in zip(maps, features):
                torch.testing.assert_close(actual, attention_map(feature))

    def test_twenty_teacher_distribution_remains_normalized_and_finite(self):
        models = [ConstantLogits([1000 if i % 2 == 0 else -1000,
                                  -1000 if i % 2 == 0 else 1000]) for i in range(20)]
        ensemble = MultiTeacherEnsemble(models, [1] * 20)
        target, _ = ensemble.distribution(torch.randn(2, 3, 32, 32))
        torch.testing.assert_close(target.exp(), torch.full((2, 2), .5))
        self.assertTrue(bool(torch.isfinite(target).all()))
        with self.assertRaises(ValueError):
            MultiTeacherEnsemble(models, [0] * 20)

    def test_kd_updates_student_and_generator_without_modifying_uploaded_teachers(self):
        teachers = [make_model('cpu'), make_model('cpu')]
        source_hashes = [model_digest(model) for model in teachers]
        ensemble = MultiTeacherEnsemble(teachers, [.25, .75])
        snapshot_hashes = ensemble.digests()
        student = make_model('cpu'); generator = ImageGenerator(output_mode='tanh')
        student_before = model_digest(student)
        generator_before = generator.project.weight.detach().clone()
        loss, loss_g, images = multiteacher_step(
            ensemble, student, generator,
            torch.optim.Adam(student.parameters(), lr=.0002),
            torch.optim.Adam(generator.parameters(), lr=.001),
            torch.randn(2, 100), student_steps=2)
        self.assertTrue(bool(torch.isfinite(torch.tensor([loss, loss_g])).all()))
        self.assertEqual(source_hashes, [model_digest(model) for model in teachers])
        self.assertEqual(snapshot_hashes, ensemble.digests())
        self.assertNotEqual(student_before, model_digest(student))
        self.assertFalse(torch.equal(generator_before, generator.project.weight))
        self.assertTrue(all(parameter.grad is None for parameter in ensemble.parameters()))
        self.assertLessEqual(float(images.abs().max()), 1)

    def test_only_active_malicious_clients_are_poisoned_after_augmentation(self):
        args = SimpleNamespace(backdoor='badnets', backdoor_poison_ratio=1,
                               backdoor_target_label=0, backdoor_patch_size=5,
                               backdoor_patch_location='upper_left', backdoor_patch_margin=0)
        data = [(torch.zeros(3, 32, 32), torch.tensor(1)) for _ in range(4)]
        for malicious in [False, True]:
            attack = build_backdoor_attack(args, 0, malicious)
            dataset = AugmentedBackdoorDataset(data, attack, augment=malicious)
            image, label = dataset[0]
            self.assertEqual(int(label), 0 if malicious else 1)
            self.assertEqual(len(dataset.poison_indices), 4 if malicious else 0)
            self.assertTrue(bool((image[:, :5, :5] == (1 if malicious else 0)).all()))
        args.backdoor_start_round = 3
        inactive = build_backdoor_attack(args, 0, True)
        self.assertEqual(len(AugmentedBackdoorDataset(data, inactive).poison_indices), 0)

    @unittest.skipUnless(shutil.which('bash'), 'launcher validation requires bash')
    def test_all_launcher_modes_construct_a_real_monitor(self):
        from flcore.security.monitor import UpdateMonitor
        script = Path(__file__).resolve().parents[1] / 'tools/reproduce_federated_zskt.sh'
        with tempfile.TemporaryDirectory() as directory:
            for mode in ['badnets', 'fedavg', 'clean']:
                environment = {k: v for k, v in os.environ.items() if not k.startswith('FL_ZSKT_')}
                environment.update(DFKD_DRY_RUN='1', DFKD_PYTHON=sys.executable)
                output = subprocess.check_output(
                    ['bash', str(script), mode], text=True, env=environment)
                preparation = shlex.split(output.splitlines()[0])
                self.assertEqual(preparation[preparation.index('--protocol') + 1], 'legacy')
                command = shlex.split(output.splitlines()[-1])
                expected = {'--local_epochs': '2', '--batch_size': '10',
                            '--local_learning_rate': '0.005', '--zskt_local_momentum': '0',
                            '--zskt_local_weight_decay': '0', '--zskt_local_augmentation': '0',
                            '--zskt_local_lr_schedule': 'constant', '--zskt_evaluation_split': 'auto',
                            '--global_rounds': '200', '--zskt_seed': '1000', '--attack_seed': '0',
                            '--backdoor_seed': '0', '--backdoor_preset': 'reference',
                            '--backdoor_poison_ratio': '0.2', '--backdoor_patch_size': '4',
                            '--backdoor_patch_location': 'lower_right', '--backdoor_patch_margin': '1'}
                for flag, value in expected.items():
                    self.assertEqual(command[command.index(flag) + 1], value)
                self.assertEqual(command[command.index('--zskt_batch_size') + 1], '128')
                interval = int(command[command.index('--attack_monitor_plot_every') + 1])
                self.assertGreater(interval, 0)
                monitor = UpdateMonitor(Path(directory) / mode, plot_every=interval)
                self.assertTrue(Path(monitor.client_csv).exists())

    def test_iid_partition_is_balanced_disjoint_and_seeded(self):
        labels = np.repeat(np.arange(10), 100)
        parts = balanced_iid_indices(labels, 20, 7)
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(1000))
        self.assertTrue(all(np.bincount(labels[part], minlength=10).tolist() == [5] * 10 for part in parts))
        again = balanced_iid_indices(labels, 20, 7)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(parts, again)))
        self.assertFalse(np.array_equal(parts[0], balanced_iid_indices(labels, 20, 8)[0]))

    def test_legacy_cache_copies_exact_records_without_repartitioning(self):
        import hashlib
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / 'source'; source.mkdir()
            output = Path(directory) / 'cached'
            (source / 'config.json').write_text(json.dumps({'num_clients': 2, 'num_classes': 10}))
            for split in ['train', 'test']:
                (source / split).mkdir()
                for client in [0, 1, 99]:
                    data = {'x': np.full((10, 3, 32, 32), -1 if split == 'train' else 1,
                                         dtype=np.float32), 'y': np.arange(10)}
                    np.savez(source / split / f'{client}.npz', data=data)
            before = (source / 'train/0.npz').read_bytes()
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare_legacy(source, output, 2)
                prepare_legacy(source, output, 2)
            self.assertEqual(manifest['train']['samples'], 20)
            self.assertEqual(manifest['test']['samples'], 20)
            self.assertEqual(before, (output / 'train/0.npz').read_bytes())
            self.assertEqual(before, (source / 'train/0.npz').read_bytes())
            self.assertFalse((output / 'train/99.npz').exists())
            self.assertEqual(manifest['train']['clients'][0]['file_sha256'],
                             hashlib.sha256(before).hexdigest())
            (source / 'train/0.npz').write_bytes(before + b'changed')
            with self.assertRaisesRegex(ValueError, 'changed'):
                prepare_legacy(source, output, 2)

    def test_client_holdout_loader_reads_only_test_and_keeps_normalized_values(self):
        from flcore.servers.serveravgzskt import load_client_holdout
        def record(dataset, index, is_train):
            self.assertFalse(is_train)
            return {'x': np.full((2, 3, 32, 32), -1 if index == 0 else 1, dtype=np.float32),
                    'y': np.array([0, 1])}
        with patch('flcore.servers.serveravgzskt.read_data', side_effect=record):
            images, labels = load_client_holdout('unused', 2, 'cpu')
        self.assertEqual(tuple(images.shape), (4, 3, 32, 32))
        self.assertEqual(float(images.min()), -1)
        self.assertEqual(float(images.max()), 1)
        self.assertEqual(len(labels), 4)

    def test_preparation_keeps_train_test_records_separate_and_refuses_overwrite(self):
        labels = np.repeat(np.arange(10), 2)
        def fake_read(root, train):
            return np.full((20, 3, 32, 32), 10 if train else 240, dtype=np.uint8), labels
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / 'dataset'
            with patch('tools.prepare_federated_zskt.read_official_split', side_effect=fake_read), \
                 contextlib.redirect_stdout(io.StringIO()):
                manifest = prepare('unused', output, 2, 0)
                prepare('unused', output, 2, 0)
                with self.assertRaises(ValueError):
                    prepare('unused', output, 2, 1)
            self.assertEqual(manifest['train']['official_samples'], 20)
            for client in range(2):
                train = np.load(output / 'train' / f'{client}.npz', allow_pickle=True)['data'].item()
                test = np.load(output / 'test' / f'{client}.npz', allow_pickle=True)['data'].item()
                self.assertLess(float(train['x'].max()), 0)
                self.assertGreater(float(test['x'].min()), 0)


class FederatedZSKTFrameworkTest(unittest.TestCase):
    """CPU-only toy fixtures: exercise main -> native clients -> FedAvg -> KD."""
    def setUp(self):
        torch.set_num_threads(4)
        self.system = Path(__file__).resolve().parents[1]

    def _run_main(self, dataset_name, output_root, cycles, resume='', interrupt=False, evaluation_split=None):
        import utils.experiment_logger as logging_module
        from flcore.servers.serveravgzskt import FedAvgZSKT
        original_logger = logging_module.setup_experiment_logger
        loggers = []
        def make_logger(args):
            logger = original_logger(args); loggers.append(logger); return logger
        argv = [str(self.system / 'main.py'), '--algorithm', 'FedAvgZSKT',
                '--device', 'cpu', '--model', 'CNN', '--dataset', dataset_name,
                '--num_clients', '2', '--join_ratio', '1', '--global_rounds', '2',
                '--local_epochs', '1', '--batch_size', '4', '--local_learning_rate', '.005',
                '--attack_malicious_ratio', '.5', '--backdoor', 'badnets',
                '--zskt_cycles_per_round', str(cycles), '--zskt_student_steps', '1',
                '--zskt_batch_size', '2', '--zskt_health_every', '1',
                '--zskt_teacher_eval_every', '1', '--attack_monitor_plot_every', '100',
                '--logger_root', str(output_root), '--goal', 'cpu_fixture']
        if resume:
            argv += ['--zskt_resume', str(resume)]
        if evaluation_split is not None:
            argv += ['--zskt_evaluation_split', evaluation_split]
        pixels = torch.arange(10 * 3 * 32 * 32).remainder(256).to(torch.uint8).reshape(10, 3, 32, 32)
        labels = torch.arange(10)
        checkpoint = FedAvgZSKT._checkpoint
        def stop_after_one(server, round_id, metrics):
            checkpoint(server, round_id, metrics)
            if interrupt and round_id == 1:
                raise InterruptedError('test interruption after a completed round')
        with patch.object(sys, 'argv', argv), \
             patch('flcore.servers.serveravgzskt.load_cifar10', return_value=(pixels, labels)), \
             patch('utils.mem_utils.MemReporter.report'), \
             patch('utils.experiment_logger.setup_experiment_logger', side_effect=make_logger), \
             patch.object(FedAvgZSKT, '_checkpoint', stop_after_one), \
             contextlib.redirect_stdout(io.StringIO()):
            try:
                if interrupt:
                    with self.assertRaises(InterruptedError):
                        runpy.run_path(str(self.system / 'main.py'), run_name='__main__')
                else:
                    runpy.run_path(str(self.system / 'main.py'), run_name='__main__')
            finally:
                for logger in reversed(loggers):
                    logger.close()
        return Path(loggers[-1].experiment_dir) / 'zskt/run0'

    def test_framework_round_order_fedavg_control_and_resume(self):
        dataset_root = self.system.parent / 'dataset'
        dataset_root.mkdir(exist_ok=True)
        with tempfile.TemporaryDirectory(prefix='Cifar10_zskt_unittest_', dir=dataset_root) as directory, \
             tempfile.TemporaryDirectory(prefix='zskt_cpu_test_') as outputs:
            dataset = Path(directory)
            rng = np.random.default_rng(17)
            for split in ['train', 'test']:
                (dataset / split).mkdir()
                for client in range(2):
                    payload = {'x': rng.uniform(-1, 1, (20, 3, 32, 32)).astype(np.float32),
                               'y': np.tile(np.arange(10), 2)}
                    np.savez(dataset / split / f'{client}.npz', data=payload)
            (dataset / 'partition_manifest.json').write_text(json.dumps({
                'source': 'official_cifar10_train_test_separate', 'num_clients': 2,
                'fixture_note': 'synthetic CPU fixture only, not a CIFAR10 experiment'}))
            full = self._run_main(dataset.name, Path(outputs) / 'full', cycles=1)
            entries = [json.loads(line) for line in (full / 'rounds.jsonl').read_text().splitlines()]
            self.assertEqual([row['round'] for row in entries], [1, 2])
            self.assertEqual(entries[-1]['kd']['student_updates'], 1)
            self.assertEqual(len(entries[-1]['malicious_uploaded_ids']), 1)
            for row in entries:
                audits = row['client_poisoning']
                self.assertEqual(len(audits), 2)
                self.assertEqual(sum(audit['poisoned_samples'] for audit in audits), 4)
                self.assertTrue(all(audit['preset'] == 'reference' for audit in audits))
            first_audit = {audit['client_id']: audit for audit in entries[0]['client_poisoning']}
            for audit in entries[1]['client_poisoning']:
                self.assertEqual(audit['poisoned_indices_sha256'],
                                 first_audit[audit['client_id']]['poisoned_indices_sha256'])
            self.assertAlmostEqual(entries[-1]['malicious_teacher_weight'], .5)
            teachers = torch.load(full / 'uploaded_teachers_latest.pt', weights_only=False)
            before = torch.load(full / 'fedavg_student_latest.pt', weights_only=False)
            for name, value in before.state_dict().items():
                expected = sum(weight * model[name] for weight, model in
                               zip(teachers['sample_weights'], teachers['models']))
                torch.testing.assert_close(value, expected)
            self.assertEqual(entries[-1]['pre_kd_student_sha256'], model_digest(before))
            self.assertEqual(entries[-1]['kd']['teacher_sha256_before'],
                             entries[-1]['kd']['teacher_sha256_after'])
            control = self._run_main(dataset.name, Path(outputs) / 'control', cycles=0)
            pre = torch.load(control / 'fedavg_student_latest.pt', weights_only=False)
            post = torch.load(control / 'global_model.pt', weights_only=False)
            self.assertEqual(model_digest(pre), model_digest(post))
            partial = self._run_main(dataset.name, Path(outputs) / 'partial', cycles=1, interrupt=True)
            resumed = self._run_main(dataset.name, Path(outputs) / 'resumed', cycles=1,
                                     resume=partial / 'checkpoint.pt')
            complete_model = torch.load(full / 'global_model.pt', weights_only=False)
            resumed_model = torch.load(resumed / 'global_model.pt', weights_only=False)
            self.assertEqual(model_digest(complete_model), model_digest(resumed_model))
            self.assertTrue((resumed / 'completed.json').exists())
            monitor = resumed.parents[1] / 'attack_monitor/run0'
            self.assertTrue((monitor / 'client_metrics.csv').exists())
            (dataset / 'partition_manifest.json').write_text(json.dumps({
                'source': 'legacy_cifar10_client_holdout', 'num_clients': 2,
                'partition_details': {'scheme': 'dirichlet', 'alpha': .1,
                                      'balanced': False, 'reused_existing_records': True},
                'fixture_note': 'normalized synthetic holdout, not a formal experiment'}))
            legacy = self._run_main(dataset.name, Path(outputs) / 'legacy', cycles=0)
            protocol = json.loads((legacy / 'protocol.json').read_text())
            self.assertEqual(protocol['data_partition'], {
                'scheme': 'dirichlet', 'alpha': .1, 'balanced': False,
                'reused_existing_records': True})
            self.assertEqual(protocol['evaluation_samples'], 40)
            self.assertIn('client test', protocol['evaluation_source'])
            self.assertFalse(protocol['local_augmentation'])
            self.assertEqual(protocol['local_momentum'], 0)
            self.assertEqual(protocol['local_weight_decay'], 0)
            self.assertEqual(protocol['local_lr_schedule'], 'constant')
            legacy_rounds = [json.loads(line) for line in (legacy / 'rounds.jsonl').read_text().splitlines()]
            self.assertTrue(all(row['local_lr'] == .005 for row in legacy_rounds))
            self.assertEqual(legacy_rounds[-1]['post_kd']['test_samples'], 40)
            with self.assertRaisesRegex(ValueError, 'Legacy training'):
                self._run_main(dataset.name, Path(outputs) / 'invalid_evaluation',
                               cycles=0, evaluation_split='official')



if __name__ == '__main__':
    unittest.main()
