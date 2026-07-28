import copy
import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import DataLoader
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from utils.data_utils import read_client_data
from utils.label_flip import LabelFlipWrapper, get_label_flip_mapping


class Client(object):
    """
    Base class for clients in federated learning.
    """

    def __init__(self, args, id, train_samples, test_samples, **kwargs):
        torch.manual_seed(0)
        self.model = copy.deepcopy(args.model)
        self.algorithm = args.algorithm
        self.dataset = args.dataset
        self.device = args.device
        self.id = id  # integer
        self.save_folder_name = args.save_folder_name

        self.num_classes = args.num_classes
        self.train_samples = train_samples
        self.test_samples = test_samples
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.local_epochs = args.local_epochs
        self.few_shot = args.few_shot
        self.attack_type = getattr(args, 'attack_type', 'none')
        self.is_malicious = kwargs.get('is_malicious', False)


        # check BatchNorm
        self.has_BatchNorm = False
        for layer in self.model.children():
            if isinstance(layer, nn.BatchNorm2d):
                self.has_BatchNorm = True
                break

        self.train_slow = kwargs['train_slow']
        self.send_slow = kwargs['send_slow']
        self.train_time_cost = {'num_rounds': 0, 'total_cost': 0.0}
        self.send_time_cost = {'num_rounds': 0, 'total_cost': 0.0}

        self.loss = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate)
        self.learning_rate_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer=self.optimizer, 
            gamma=args.learning_rate_decay_gamma
        )
        self.learning_rate_decay = args.learning_rate_decay


    def load_train_data(self, batch_size=None, drop_last=None):
        """Build a training loader.

        Ordinary local-epoch algorithms keep the historical ``drop_last``
        behavior.  FedSGD opts out by default so a client with fewer samples
        than the configured global batch size can still contribute one
        gradient and is represented in its training metrics.
        """
        if batch_size == None:
            batch_size = self.batch_size
        if drop_last is None:
            drop_last = self.algorithm != "FedSGD"
        train_data = read_client_data(self.dataset, self.id, is_train=True, few_shot=self.few_shot)
        if (self.is_malicious and self.attack_type == 'label_flip'
                and self.dataset.upper().startswith('MNIST')):
            train_data = LabelFlipWrapper(train_data, self.dataset, self.attack_type)
        return DataLoader(train_data, batch_size, drop_last=drop_last, shuffle=True)


    def load_test_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        test_data = read_client_data(self.dataset, self.id, is_train=False, few_shot=self.few_shot)
        return DataLoader(test_data, batch_size, drop_last=False, shuffle=True)
        
    def set_parameters(self, model):
        """Synchronize the complete global model state to this client.

        BatchNorm running statistics are registered buffers rather than
        parameters.  Copying only ``parameters()`` left every client with
        stale local BN statistics after the server aggregated new ones.
        """
        source_parameters = dict(model.named_parameters())
        target_parameters = dict(self.model.named_parameters())
        if set(source_parameters) != set(target_parameters):
            raise ValueError("Global and client models do not have matching parameters")
        source_buffers = dict(model.named_buffers())
        target_buffers = dict(self.model.named_buffers())
        if set(source_buffers) != set(target_buffers):
            raise ValueError("Global and client models do not have matching buffers")

        with torch.no_grad():
            for name, target_parameter in target_parameters.items():
                target_parameter.copy_(source_parameters[name].detach())
            for name, target_buffer in target_buffers.items():
                target_buffer.copy_(source_buffers[name].detach())

    def clone_model(self, model, target):
        source_parameters = dict(model.named_parameters())
        target_parameters = dict(target.named_parameters())
        source_buffers = dict(model.named_buffers())
        target_buffers = dict(target.named_buffers())
        if (set(source_parameters) != set(target_parameters)
                or set(source_buffers) != set(target_buffers)):
            raise ValueError("Models do not have matching state for cloning")
        with torch.no_grad():
            for name, target_parameter in target_parameters.items():
                target_parameter.copy_(source_parameters[name].detach())
            for name, target_buffer in target_buffers.items():
                target_buffer.copy_(source_buffers[name].detach())
            # target_param.grad intentionally remains unset.

    def update_parameters(self, model, new_params):
        for param, new_param in zip(model.parameters(), new_params):
            param.data = new_param.data.clone()

    def test_metrics(self):
        testloaderfull = self.load_test_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        test_acc = 0
        test_num = 0
        y_prob = []
        y_true = []
        # for SA/ASR
        src_total = 0
        src_correct = 0
        src_to_tgt = 0
        src_label, tgt_label = get_label_flip_mapping(self.dataset)
        
        with torch.no_grad():
            for x, y in testloaderfull:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)

                preds = torch.argmax(output, dim=1)
                test_acc += (torch.sum(preds == y)).item()
                test_num += y.shape[0]

                # SA/ASR counting when mapping exists
                if src_label is not None and tgt_label is not None:
                    mask = (y == src_label)
                    if mask.any():
                        src_total += int(mask.sum().item())
                        src_correct += int((preds[mask] == y[mask]).sum().item())
                        src_to_tgt += int((preds[mask] == tgt_label).sum().item())

                y_prob.append(output.detach().cpu().numpy())
                nc = self.num_classes
                if self.num_classes == 2:
                    nc += 1
                lb = label_binarize(y.detach().cpu().numpy(), classes=np.arange(nc))
                if self.num_classes == 2:
                    lb = lb[:, :2]
                y_true.append(lb)

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        y_prob = np.concatenate(y_prob, axis=0)
        y_true = np.concatenate(y_true, axis=0)

        auc = metrics.roc_auc_score(y_true, y_prob, average='micro')
        
        return test_acc, test_num, auc, src_total, src_correct, src_to_tgt

    def train_metrics(self):
        trainloader = self.load_train_data()
        # self.model = self.load_model('model')
        # self.model.to(self.device)
        self.model.eval()

        train_num = 0
        losses = 0
        with torch.no_grad():
            for x, y in trainloader:
                if type(x) == type([]):
                    x[0] = x[0].to(self.device)
                else:
                    x = x.to(self.device)
                y = y.to(self.device)
                output = self.model(x)
                loss = self.loss(output, y)
                train_num += y.shape[0]
                losses += loss.item() * y.shape[0]

        # self.model.cpu()
        # self.save_model(self.model, 'model')

        return losses, train_num

    # def get_next_train_batch(self):
    #     try:
    #         # Samples a new batch for persionalizing
    #         (x, y) = next(self.iter_trainloader)
    #     except StopIteration:
    #         # restart the generator if the previous generator is exhausted.
    #         self.iter_trainloader = iter(self.trainloader)
    #         (x, y) = next(self.iter_trainloader)

    #     if type(x) == type([]):
    #         x = x[0]
    #     x = x.to(self.device)
    #     y = y.to(self.device)

    #     return x, y


    def save_item(self, item, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        if not os.path.exists(item_path):
            os.makedirs(item_path)
        torch.save(item, os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    def load_item(self, item_name, item_path=None):
        if item_path == None:
            item_path = self.save_folder_name
        return torch.load(os.path.join(item_path, "client_" + str(self.id) + "_" + item_name + ".pt"))

    # @staticmethod
    # def model_exists():
    #     return os.path.exists(os.path.join("models", "server" + ".pt"))
