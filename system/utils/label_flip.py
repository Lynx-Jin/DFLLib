import torch
from torch.utils.data import Dataset


def is_mnist_variant(dataset_name: str) -> bool:
    """Return whether a configured dataset name is an MNIST-derived split."""
    return isinstance(dataset_name, str) and dataset_name.upper().startswith('MNIST')


class LabelFlipWrapper(Dataset):
    def __init__(self, base_dataset, dataset_name: str, attack_type: str):
        self.base = base_dataset
        self.dataset_name = dataset_name
        self.attack_type = attack_type

    def __len__(self):
        return len(self.base)

    def __getitem__(self, index):
        x, y = self.base[index]
        if self.attack_type == 'label_flip' and is_mnist_variant(self.dataset_name):
            if isinstance(y, torch.Tensor):
                if y.dim() == 0:
                    if int(y.item()) == 7:
                        y = torch.as_tensor(1, dtype=y.dtype, device=y.device)
                else:
                    pass
            else:
                try:
                    if int(y) == 7:
                        try:
                            y = type(y)(1)
                        except Exception:
                            y = 1
                except Exception:
                    pass
        return x, y

def get_label_flip_mapping(dataset_name: str):
    if is_mnist_variant(dataset_name):
        return 7, 1
    return None, None
