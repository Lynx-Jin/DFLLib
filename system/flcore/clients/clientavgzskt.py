"""Local CNN training shared by the ZSKT arm and its FedAvg-only control."""
import hashlib
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from flcore.clients.clientavg import clientAVG
from utils.data_utils import read_client_data


class AugmentedBackdoorDataset(Dataset):
    def __init__(self, data, attack, augment=False, apply_backdoor=True):
        self.data = data
        self.attack = attack
        # Select indices from raw records; augmentation must not run during selection.
        self.poison_indices = (attack.choose_poison_indices(data, attack.round_id)
                               if apply_backdoor and attack.is_malicious and attack.active() else set())
        self.transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4, fill=-1),
            transforms.RandomHorizontalFlip(),
        ]) if augment else None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        images, label = self.data[index]
        images = images.clone()
        if self.transform is not None:
            images = self.transform(images)
        if index in self.poison_indices:
            images, label = self.attack.poison_training_example(images, label)
        return images, label


class clientAVGZSKT(clientAVG):
    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        super().__init__(args, id, train_samples, test_samples, **kwargs)
        self.augment_images = bool(args.zskt_local_augmentation)
        self._poison_protocol_logged = False
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate,
                                         momentum=args.zskt_local_momentum,
                                         weight_decay=args.zskt_local_weight_decay)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=args.learning_rate_decay_gamma)

    def load_train_data(self, batch_size=None, drop_last=None, apply_backdoor=True):
        data = read_client_data(self.dataset, self.id, is_train=True,
                                few_shot=self.few_shot)
        data = AugmentedBackdoorDataset(data, self.backdoor, self.augment_images,
                                         apply_backdoor)
        selected = np.asarray(sorted(data.poison_indices), dtype=np.int64)
        eligible = sum(self.backdoor._eligible_label(int(label)) for _, label in data.data)
        self.last_poison_audit = {
            'client_id': self.id, 'round': self.backdoor.round_id,
            'malicious': self.is_malicious, 'preset': self.backdoor.spec.preset,
            'samples': len(data), 'eligible_samples': eligible,
            'poisoned_samples': len(selected),
            'poisoned_indices_sha256': hashlib.sha256(selected.tobytes()).hexdigest(),
            'poison_ratio_basis': ('all_local_samples' if self.backdoor.spec.preset == 'reference'
                                   else 'eligible_samples'),
            'index_schedule': ('fixed_per_client' if self.backdoor.spec.preset == 'reference'
                               else 'resampled_per_round'),
        }
        if self.is_malicious and not self._poison_protocol_logged:
            print(f'Backdoor client {self.id}: preset={self.backdoor.spec.preset}, '
                  f'poisoned={len(selected)}/{len(data)}, '
                  f'indices={self.last_poison_audit["poisoned_indices_sha256"]}', flush=True)
            self._poison_protocol_logged = True
        return DataLoader(data, batch_size or self.batch_size, shuffle=True,
                          drop_last=True if drop_last is None else drop_last)

    def train(self):
        # Local momentum is fresh after every global broadcast in both arms.
        self.optimizer.state.clear()
        return super().train()
