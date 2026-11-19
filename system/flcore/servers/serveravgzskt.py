"""FedAvg followed by generator-only distillation from uploaded CNN teachers."""
import copy
from dataclasses import asdict
import csv
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch

from flcore.clients.clientavgzskt import clientAVGZSKT
from flcore.servers.serverbase import Server
from flcore.security.backdoor import build_backdoor_attack
from utils.data_utils import read_data
from flcore.trainmodel.models import FedAvgCNN
from flcore.trainmodel.multiteacher_zskt import MultiTeacherEnsemble, multiteacher_step
from tools.run_datafree_badnets import (ImageGenerator, evaluate, load_cifar10,
                                       model_digest, student_is_collapsed,
                                       synthetic_health, write_json)


def load_client_holdout(dataset, num_clients, device):
    """Use the legacy held-out client records without normalizing twice."""
    images, labels = [], []
    for client_id in range(num_clients):
        payload = read_data(dataset, client_id, is_train=False)
        x = torch.as_tensor(payload['x'], dtype=torch.float32)
        y = torch.as_tensor(payload['y'], dtype=torch.long)
        if x.ndim != 4 or tuple(x.shape[1:]) != (3, 32, 32) or len(x) != len(y):
            raise ValueError('Unexpected held-out client image shape')
        if not len(x) or float(x.min()) < -1.001 or float(x.max()) > 1.001:
            raise ValueError('Client images must already be normalized to [-1,1]')
        if not bool(((y >= 0) & (y < 10)).all()):
            raise ValueError('Unexpected CIFAR10 labels')
        images.append(x)
        labels.append(y)
    return torch.cat(images).to(device), torch.cat(labels).to(device)


class FedAvgZSKT(Server):
    def __init__(self, args, times):
        np.random.seed(args.zskt_seed + times)
        random.seed(args.zskt_seed + times)
        torch.set_num_threads(args.zskt_threads)
        if torch.device(args.device).type == "cuda":
            torch.backends.cudnn.benchmark = True
        super().__init__(args, times)
        if not isinstance(self.global_model, FedAvgCNN) or args.num_classes != 10:
            raise ValueError("FedAvgZSKT currently supports the full CIFAR10 FedAvgCNN only")
        self.set_slow_clients()
        self.set_clients(clientAVGZSKT)
        self.zskt_dir = Path(self.experiment_log_dir) / 'zskt' / f'run{times}'
        self.zskt_dir.mkdir(parents=True, exist_ok=True)
        if self.backdoor_evaluator is not None:
            self.eval_attack = self.backdoor_evaluator.attack
        else:
            evaluation_args = copy.copy(args)
            evaluation_args.backdoor = 'badnets'
            self.eval_attack = build_backdoor_attack(evaluation_args, None, False, evaluation=True)
        manifest_path = Path('../dataset') / args.dataset / 'partition_manifest.json'
        if not manifest_path.is_file():
            raise ValueError('Prepare the recorded client dataset with prepare_federated_zskt.py')
        manifest = json.loads(manifest_path.read_text())
        source = manifest.get('source')
        if (source not in {'official_cifar10_train_test_separate', 'legacy_cifar10_client_holdout'}
                or manifest.get('num_clients') != self.num_clients):
            raise ValueError('Client partitions need an official-split or legacy-holdout manifest')
        requested = args.zskt_evaluation_split
        if source == 'legacy_cifar10_client_holdout' and requested == 'official':
            raise ValueError('Legacy training may include official test records; use its client holdout')
        self.test_input_normalized = (requested == 'clients' or
            (requested == 'auto' and source == 'legacy_cifar10_client_holdout'))
        if self.test_input_normalized:
            self.test_images, self.test_labels = load_client_holdout(
                args.dataset, self.num_clients, self.device)
            evaluation_source = 'pooled held-out client test records; normalized [-1,1]'
        else:
            self.test_images, self.test_labels = load_cifar10(
                args.zskt_raw_data_root, False, self.device)
            evaluation_source = 'official CIFAR10 test'

        partition_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        torch.manual_seed(args.zskt_seed + times)
        self.cuda_devices = ([torch.device(self.device).index or 0]
                             if torch.device(self.device).type == 'cuda' else [])
        # Keep KD RNG independent from client shuffling/augmentation across arms.
        with torch.random.fork_rng(devices=self.cuda_devices):
            torch.manual_seed(args.zskt_seed + 10000 + times)
            self.generator = ImageGenerator(output_mode='tanh').to(self.device)
        self.generator_optimizer = torch.optim.Adam(
            self.generator.parameters(), lr=args.zskt_generator_lr)
        self.generator_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.generator_optimizer, max(1, args.global_rounds * args.zskt_cycles_per_round))
        self.protocol = {
            'algorithm': 'FedAvg -> probability-ensemble ZSKT -> next broadcast',
            'dataset': args.dataset, 'num_clients': self.num_clients,
            'partition_manifest_sha256': partition_digest,
            'data_protocol': source,
            'evaluation_source': evaluation_source,
            'evaluation_samples': len(self.test_labels),
            'join_ratio': self.join_ratio, 'global_rounds': args.global_rounds,
            'local_epochs': self.local_epochs, 'local_lr': self.learning_rate,
            'local_momentum': args.zskt_local_momentum,
            'local_weight_decay': args.zskt_local_weight_decay,
            'local_augmentation': bool(args.zskt_local_augmentation),
            'local_lr_min': args.zskt_local_lr_min,
            'local_lr_schedule': args.zskt_local_lr_schedule,
            'local_batch_size': self.batch_size,
            'local_drop_last': True,
            'malicious_ids': self.malicious_ids,
            'backdoor_spec': asdict(self.eval_attack.spec),
            'backdoor_preset': self.eval_attack.spec.preset,
            'poison_ratio': args.backdoor_poison_ratio,
            'poison_ratio_basis': ('all_local_samples' if self.eval_attack.spec.preset == 'reference'
                                   else 'eligible_samples'),
            'poison_index_schedule': ('fixed_per_client' if self.eval_attack.spec.preset == 'reference'
                                      else 'resampled_per_round'),
            'backdoor': args.backdoor, 'patch_size': args.backdoor_patch_size,
            'patch_location': args.backdoor_patch_location,
            'patch_margin': args.backdoor_patch_margin,
            'target_label': args.backdoor_target_label,
            'cycles_per_round': args.zskt_cycles_per_round,
            'student_steps': args.zskt_student_steps,
            'student_lr': args.zskt_student_lr, 'generator_lr': args.zskt_generator_lr,
            'temperature': args.zskt_temperature, 'attention_weight': args.zskt_attention_weight,
            'batch_size': args.zskt_batch_size, 'seed': args.zskt_seed,
            'teacher_pool': 'received client snapshots; sample-count probability weighting',
            'student_initialization_each_round': 'exact FedAvg aggregation of received models',
            'generator_lifetime': 'one Tanh generator across rounds, persistent Adam and global cosine',
            'student_optimizer': 'fresh Adam and per-round cosine after each aggregation',
            'server_distillation_real_training_samples': 0,
            'server_distillation_original_labels': 0,
            'server_distillation_injected_triggers': 0,
            'evaluation': evaluation_source + '; no-grad, never used by KD loss',
            'attribution': 'final ASR alone cannot separate FedAvg inheritance from KD; compare pre/post and FedAvg-only arm',
        }
        if manifest.get('partition_details') is not None:
            self.protocol['data_partition'] = manifest['partition_details']
        write_json(self.zskt_dir / 'protocol.json', self.protocol)
        write_json(self.zskt_dir / 'command.json', {
            'executable': sys.executable, 'argv': sys.argv, 'cwd': str(Path.cwd()),
            'torch_version': torch.__version__,
        })
        source_root = Path(__file__).resolve().parents[2]
        source_hashes = {}
        for relative in ['main.py', 'flcore/servers/serveravgzskt.py',
                         'flcore/clients/clientavgzskt.py',
                         'flcore/trainmodel/multiteacher_zskt.py',
                         'flcore/trainmodel/models.py', 'tools/run_datafree_badnets.py',
                         'flcore/security/backdoor/core.py', 'flcore/security/backdoor/__init__.py',
                         'tools/prepare_federated_zskt.py', 'tools/reproduce_federated_zskt.sh']:
            content = (source_root / relative).read_bytes()
            target = self.zskt_dir / 'source' / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(content)
            source_hashes[relative] = hashlib.sha256(content).hexdigest()
        write_json(self.zskt_dir / 'source_sha256.json', source_hashes)

        self.first_round = 1
        self.Budget = []
        if args.zskt_resume:
            self._resume(Path(args.zskt_resume))
        print(f'FedAvgZSKT: clients={self.num_clients}, malicious={self.malicious_ids}, '
              f'KD cycles/round={args.zskt_cycles_per_round}; logs={self.zskt_dir}')

    def _evaluate(self, model):
        return evaluate(model, self.test_images, self.test_labels, self.eval_attack,
                        input_normalized=self.test_input_normalized)

    def _metric(self, round_id, stage, metrics):
        row = {'round': round_id, 'stage': stage, **{
            key: value for key, value in metrics.items() if not isinstance(value, list)}}
        path = self.zskt_dir / 'metrics.csv'
        exists = path.exists()
        with path.open('a', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        print(f'[round {round_id} {stage}] Acc={metrics["clean_accuracy"]:.4f} '
              f'ASR={metrics["asr"]:.4f} clean_target={metrics["clean_target_rate"]:.4f}', flush=True)

    def _distill(self, ensemble, round_id):
        args = self.args
        teacher_before = ensemble.digests()
        student_optimizer = torch.optim.Adam(self.global_model.parameters(),
                                              lr=args.zskt_student_lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            student_optimizer, args.zskt_cycles_per_round)
        collapsed = 0
        last_health = None
        started = time.monotonic()
        with torch.random.fork_rng(devices=self.cuda_devices):
            torch.manual_seed(args.zskt_seed + 1000003 * round_id + self.times)
            for cycle in range(1, args.zskt_cycles_per_round + 1):
                noise = torch.randn(args.zskt_batch_size, 100, device=self.device)
                loss, loss_g, images = multiteacher_step(
                    ensemble, self.global_model, self.generator, student_optimizer,
                    self.generator_optimizer, noise, args.zskt_student_steps,
                    args.zskt_temperature, args.zskt_attention_weight)
                scheduler.step()
                self.generator_scheduler.step()
                if cycle % args.zskt_health_every == 0 or cycle == args.zskt_cycles_per_round:
                    last_health = synthetic_health(ensemble, self.global_model, images)
                    collapsed = collapsed + 1 if student_is_collapsed(last_health) else 0
                    detail = {'round': round_id, 'cycle': cycle, 'student_loss': loss,
                              'generator_loss': loss_g, 'health': last_health,
                              'elapsed_seconds': time.monotonic() - started}
                    write_json(self.zskt_dir / 'kd_progress.json', detail)
                    print(f'[round {round_id} KD {cycle}/{args.zskt_cycles_per_round}] '
                          f'loss={loss:.5f}, G={loss_g:.5f}', flush=True)
                    if args.zskt_collapse_patience > 0 and collapsed >= args.zskt_collapse_patience:
                        write_json(self.zskt_dir / 'training_failure.json', detail)
                        torch.save(self.global_model.state_dict(),
                                   self.zskt_dir / 'collapsed_student.pt')
                        raise RuntimeError('Server CNN collapsed; inspect training_failure.json. '
                                           'Last completed-round checkpoint is preserved.')
        teacher_after = ensemble.digests()
        if teacher_before != teacher_after:
            raise RuntimeError('A frozen uploaded teacher changed during ZSKT')
        return {'cycles': args.zskt_cycles_per_round,
                'student_updates': args.zskt_cycles_per_round * args.zskt_student_steps,
                'student_loss': loss, 'generator_loss': loss_g, 'health': last_health,
                'teacher_sha256_before': teacher_before, 'teacher_sha256_after': teacher_after,
                'server_training_real_samples': 0, 'server_injected_triggers': 0}

    def _checkpoint(self, round_id, metrics):
        state = {'round': round_id, 'protocol': self.protocol,
                 'global_model': self.global_model.state_dict(),
                 'generator': self.generator.state_dict(),
                 'generator_optimizer': self.generator_optimizer.state_dict(),
                 'generator_scheduler': self.generator_scheduler.state_dict(),
                 'torch_rng': torch.get_rng_state(),
                 'cuda_rng': torch.cuda.get_rng_state_all() if self.cuda_devices else [],
                 'numpy_rng': np.random.get_state(), 'python_rng': random.getstate(),
                 'rs_test_acc': self.rs_test_acc, 'rs_ASR': self.rs_ASR,
                 'metrics': metrics}
        temporary = self.zskt_dir / 'checkpoint.tmp'
        torch.save(state, temporary)
        temporary.replace(self.zskt_dir / 'checkpoint.pt')
        torch.save(copy.deepcopy(self.global_model).cpu(), self.zskt_dir / 'global_model.pt')
        torch.save({'round': round_id, 'client_ids': self.uploaded_ids,
                    'sample_weights': self.uploaded_weights,
                    'models': [{key: value.detach().cpu() for key, value in model.state_dict().items()}
                               for model in self.uploaded_models]},
                   self.zskt_dir / 'uploaded_teachers_latest.pt')

    def _resume(self, path):
        saved = torch.load(path, map_location=self.device, weights_only=False)
        if saved['protocol'] != self.protocol:
            raise ValueError('Resume requires identical protocol, client split and total budget')
        self.global_model.load_state_dict(saved['global_model'])
        self.generator.load_state_dict(saved['generator'])
        self.generator_optimizer.load_state_dict(saved['generator_optimizer'])
        self.generator_scheduler.load_state_dict(saved['generator_scheduler'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if self.cuda_devices:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        np.random.set_state(saved['numpy_rng'])
        random.setstate(saved['python_rng'])
        self.rs_test_acc = saved['rs_test_acc']
        self.rs_ASR = saved['rs_ASR']
        self.first_round = saved['round'] + 1

    def train(self):
        args = self.args
        if self.first_round == 1:
            self._metric(0, 'initial', self._evaluate(self.global_model))
        for round_id in range(self.first_round, self.global_rounds + 1):
            started = time.monotonic()
            self.update_monitor_round = round_id
            self.selected_clients = self.select_clients()
            self.send_models()
            lr = self.learning_rate
            if args.zskt_local_lr_schedule == 'cosine':
                lr = args.zskt_local_lr_min + 0.5 * (
                    self.learning_rate - args.zskt_local_lr_min) * (
                        1 + math.cos(math.pi * (round_id - 1) / self.global_rounds))
            for client in self.selected_clients:
                for group in client.optimizer.param_groups:
                    group['lr'] = lr
                client.train()
            self.receive_models()
            # Clone/uploaded teacher snapshots are preserved independently of student updates.
            ensemble = MultiTeacherEnsemble(self.uploaded_models, self.uploaded_weights)
            submitted_hashes = [model_digest(model) for model in self.uploaded_models]
            self.aggregate_parameters()
            pre_hash = model_digest(self.global_model)
            pre = self._evaluate(self.global_model)
            self._metric(round_id, 'fedavg_before_kd', pre)
            torch.save(copy.deepcopy(self.global_model).cpu(),
                       self.zskt_dir / 'fedavg_student_latest.pt')
            ensemble_metrics = self._evaluate(ensemble)
            self._metric(round_id, 'teacher_ensemble', ensemble_metrics)
            individual = []
            if round_id == 1 or round_id % args.zskt_teacher_eval_every == 0 or round_id == self.global_rounds:
                for client_id, weight, teacher in zip(self.uploaded_ids, self.uploaded_weights,
                                                       ensemble.models):
                    individual.append({'client_id': client_id,
                                       'malicious': client_id in self.malicious_ids,
                                       'sample_weight': weight,
                                       'metrics': self._evaluate(teacher)})
            kd = self._distill(ensemble, round_id) if args.zskt_cycles_per_round > 0 else {
                'cycles': 0, 'student_updates': 0, 'server_training_real_samples': 0}
            if submitted_hashes != [model_digest(model) for model in self.uploaded_models]:
                raise RuntimeError('Client submissions changed during server distillation')
            post = self._evaluate(self.global_model)
            self._metric(round_id, 'after_kd' if args.zskt_cycles_per_round else 'no_kd_control', post)
            row = {'round': round_id, 'uploaded_client_ids': self.uploaded_ids,
                   'sample_weights': self.uploaded_weights,
                   'malicious_uploaded_ids': [i for i in self.uploaded_ids if i in self.malicious_ids],
                   'malicious_teacher_weight': sum(w for i, w in zip(self.uploaded_ids, self.uploaded_weights)
                                                    if i in self.malicious_ids),
                   'client_poisoning': [getattr(self.clients[cid], 'last_poison_audit', None)
                                        for cid in self.uploaded_ids],
                   'local_lr': lr, 'pre_kd_student_sha256': pre_hash,
                   'pre_kd': pre, 'teacher_ensemble': ensemble_metrics,
                   'post_kd': post, 'delta_asr': post['asr'] - pre['asr'],
                   'delta_clean_accuracy': post['clean_accuracy'] - pre['clean_accuracy'],
                   'teachers': individual, 'kd': kd,
                   'elapsed_seconds': time.monotonic() - started}
            with (self.zskt_dir / 'rounds.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(row, allow_nan=False) + '\n')
            write_json(self.zskt_dir / 'status.json', row)
            self.rs_test_acc.append(post['clean_accuracy'])
            self.rs_ASR.append(post['asr'])
            self._checkpoint(round_id, post)
            self.Budget.append(time.monotonic() - started)
            print(f'Round {round_id}/{self.global_rounds} finished in {self.Budget[-1]:.1f}s', flush=True)
        self.save_results()
        self.save_global_model()
        write_json(self.zskt_dir / 'completed.json', {'rounds': self.global_rounds,
                   'last_round': self.first_round - 1 if not self.Budget else self.global_rounds,
                   'final_clean_accuracy': self.rs_test_acc[-1],
                   'final_asr': self.rs_ASR[-1]})


def add_federated_zskt_args(parser):
    group = parser.add_argument_group('FedAvgZSKT: multi-teacher generator distillation')
    group.add_argument('--zskt_cycles_per_round', type=int, default=200)
    group.add_argument('--zskt_student_steps', type=int, default=10)
    group.add_argument('--zskt_batch_size', type=int, default=128)
    group.add_argument('--zskt_student_lr', type=float, default=0.0002)
    group.add_argument('--zskt_generator_lr', type=float, default=0.001)
    group.add_argument('--zskt_temperature', type=float, default=1)
    group.add_argument('--zskt_attention_weight', type=float, default=250)
    group.add_argument('--zskt_raw_data_root', default='../dataset/Cifar10/rawdata')
    group.add_argument('--zskt_teacher_eval_every', type=int, default=10)
    group.add_argument('--zskt_health_every', type=int, default=50)
    group.add_argument('--zskt_collapse_patience', type=int, default=3)
    group.add_argument('--zskt_local_momentum', type=float, default=0)
    group.add_argument('--zskt_local_weight_decay', type=float, default=0)
    group.add_argument('--zskt_local_augmentation', type=int, choices=(0, 1), default=0)
    group.add_argument('--zskt_local_lr_min', type=float, default=0.0001)
    group.add_argument('--zskt_local_lr_schedule', choices=('constant', 'cosine'), default='constant')
    group.add_argument('--zskt_evaluation_split', choices=('auto', 'clients', 'official'), default='auto')
    group.add_argument('--zskt_seed', type=int, default=0)
    group.add_argument('--zskt_threads', type=int, default=4)
    group.add_argument('--zskt_resume', default='', help='Completed-round checkpoint.pt; same total budget')


def validate_federated_zskt_options(args):
    if args.algorithm != 'FedAvgZSKT':
        return
    if args.model != 'CNN' or 'Cifar10' not in args.dataset or args.num_classes != 10:
        raise ValueError('FedAvgZSKT requires CNN and a Cifar10 dataset prepared by its helper')
    if args.attack_type != 'none' or args.defense_type != 'none':
        raise ValueError('This FedAvgZSKT experiment requires no update attack and no defense')
    if args.backdoor not in {'none', 'badnets'}:
        raise ValueError('This FedAvgZSKT experiment implements local BadNets only')
    if args.global_rounds < 1 or args.num_clients < 2 or args.local_epochs < 1:
        raise ValueError('Require positive rounds/local epochs and at least two clients')
    if args.learning_rate_decay:
        raise ValueError('FedAvgZSKT uses --zskt_local_lr_schedule; keep legacy --learning_rate_decay disabled')
    if args.zskt_cycles_per_round < 0 or args.zskt_collapse_patience < 0:
        raise ValueError('KD cycle count and collapse patience must be nonnegative')
    if min(args.zskt_student_steps, args.zskt_batch_size, args.zskt_teacher_eval_every,
           args.zskt_health_every, args.zskt_threads) < 1 or args.zskt_batch_size < 2:
        raise ValueError('KD/health counts must be positive; G BatchNorm requires batch size >=2')
    values = [args.zskt_student_lr, args.zskt_generator_lr, args.zskt_temperature,
              args.local_learning_rate]
    if any(not math.isfinite(value) or value <= 0 for value in values):
        raise ValueError('Learning rates and temperature must be finite and positive')
    values = [args.zskt_attention_weight, args.zskt_local_weight_decay,
              args.zskt_local_lr_min, args.zskt_local_momentum]
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError('Regularization and local schedule settings must be finite/nonnegative')
    if args.zskt_local_momentum >= 1 or args.zskt_local_lr_min > args.local_learning_rate:
        raise ValueError('Require local momentum <1 and minimum local LR <= initial LR')
    if getattr(args, 'attack_monitor_plot_every', 10) <= 0:
        raise ValueError('--attack_monitor_plot_every must be positive (0 does not disable monitoring)')
    if args.zskt_seed < 0 or not 0 < args.join_ratio <= 1:
        raise ValueError('Require nonnegative ZSKT seed and join ratio in (0,1]')
