"""One-minibatch gradient client for synchronous FedSGD.

The client stores ``global_parameter + gradient`` in ``self.model`` as a
gradient carrier.  This keeps the existing attack/monitor interface (which
observes ``client_model - global_model``) in gradient coordinates rather than
silently treating a model delta as a gradient.
"""

import time

import torch

from flcore.clients.clientbase import Client


class clientFedSGD(Client):
    """Compute exactly one stochastic minibatch gradient per FL round."""

    def train(self):
        # Keep ``self.batch_size`` as the global upper bound.  Unlike local
        # FedAvg epochs, FedSGD must not discard an entire client merely
        # because its local dataset is smaller than that upper bound: with
        # drop_last=False the first (and only) batch contains all local data.
        trainloader = self.load_train_data(drop_last=False)
        try:
            x, y = next(iter(trainloader))
        except StopIteration as error:
            raise RuntimeError(
                f"Client {self.id} has no training samples; regenerate the dataset "
                "or exclude empty clients from this experiment.") from error

        self.model.train()
        start_time = time.time()
        if type(x) == type([]):
            x[0] = x[0].to(self.device)
        else:
            x = x.to(self.device)
        y = y.to(self.device)

        self.model.zero_grad(set_to_none=True)
        loss = self.loss(self.model(x), y)
        loss.backward()

        # The server interprets model - global_model as the uploaded gradient.
        # Do not call the inherited local SGD optimizer here: FedSGD applies the
        # selected server optimizer exactly once to the weighted mean gradient.
        with torch.no_grad():
            for parameter in self.model.parameters():
                if parameter.grad is None:
                    parameter.copy_(parameter)
                else:
                    parameter.add_(parameter.grad)

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
