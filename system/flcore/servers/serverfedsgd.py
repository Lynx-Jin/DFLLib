"""Synchronous one-minibatch FedSGD with a server-side SGD or Adam step."""

import time

import torch

from flcore.clients.clientfedsgd import clientFedSGD
from flcore.servers.serverbase import Server


class FedSGD(Server):
    """Aggregate one client gradient per round, then update the global model."""

    def __init__(self, args, times):
        super().__init__(args, times)
        self.set_slow_clients()
        self.set_clients(clientFedSGD)
        self.server_optimizer = self._build_server_optimizer(args)
        print(f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating FedSGD server and clients.")
        self.Budget = []

    def _build_server_optimizer(self, args):
        name = args.fedsgd_optimizer.lower()
        learning_rate = args.fedsgd_learning_rate
        if name == "sgd":
            return torch.optim.SGD(
                self.global_model.parameters(),
                lr=learning_rate,
                momentum=args.fedsgd_momentum,
                weight_decay=args.fedsgd_weight_decay,
            )
        if name == "adam":
            return torch.optim.Adam(
                self.global_model.parameters(),
                lr=learning_rate,
                betas=(args.fedsgd_adam_beta1, args.fedsgd_adam_beta2),
                eps=args.fedsgd_adam_eps,
                weight_decay=args.fedsgd_weight_decay,
            )
        raise ValueError(f"Unsupported FedSGD optimizer: {args.fedsgd_optimizer}")

    def aggregate_parameters(self):
        """Apply the chosen server optimizer to the weighted mean gradient."""
        if getattr(self, 'skip_aggregation', False):
            return
        assert len(self.uploaded_models) > 0
        self.server_optimizer.zero_grad(set_to_none=True)
        with torch.no_grad():
            for global_parameter in self.global_model.parameters():
                global_parameter.grad = torch.zeros_like(global_parameter)
            if self.defense_aggregated_model is not None:
                # Coordinate-wise defenses return global + robust aggregate gradient.
                for global_parameter, aggregate_parameter in zip(
                        self.global_model.parameters(), self.defense_aggregated_model.parameters()):
                    global_parameter.grad.copy_(
                        aggregate_parameter.detach() - global_parameter.detach())
                self.defense_aggregated_model = None
            else:
                for weight, gradient_carrier in zip(
                        self.uploaded_weights, self.uploaded_models):
                    for global_parameter, carrier_parameter in zip(
                            self.global_model.parameters(), gradient_carrier.parameters()):
                        # clientFedSGD encodes the stochastic gradient as carrier - global.
                        global_parameter.grad.add_(
                            weight * (carrier_parameter.detach() - global_parameter.detach()))
        self.server_optimizer.step()
        # Gradients update parameters, but client forward passes also update
        # BatchNorm buffers.  Synchronize those buffers after the server step.
        self._aggregate_model_buffers(
            self.global_model, self.uploaded_models, self.uploaded_weights)

    def train(self):
        for round_id in range(self.global_rounds + 1):
            start_time = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()

            if round_id % self.eval_gap == 0:
                print(f"\n-------------Round number: {round_id}-------------")
                print("\nEvaluate global model")
                self.evaluate()

            for client in self.selected_clients:
                client.train()

            self.receive_models()
            if self.dlg_eval and round_id % self.dlg_gap == 0:
                self.call_dlg(round_id)
            self.aggregate_parameters()

            self.Budget.append(time.time() - start_time)
            print('-' * 25, 'time cost', '-' * 25, self.Budget[-1])
            if self.auto_break and self.check_done(
                    acc_lss=[self.rs_test_acc], top_cnt=self.top_cnt):
                break

        print("\nBest accuracy.")
        print(max(self.rs_test_acc))
        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:]) / len(self.Budget[1:]))
        self.save_results()
        self.save_global_model()
