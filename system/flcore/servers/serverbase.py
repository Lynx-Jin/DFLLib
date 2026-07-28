import torch
import os
import numpy as np
import h5py
import copy
import time
import random
from utils.data_utils import read_client_data
from utils.dlg import DLG
from flcore.security.attack import FangKrumAttack, MinMaxMinSumAttack
from flcore.security.defense import (
    BulyanDefense, CoordinateDefense, CosineClusterDefense, DnCDefense,
    KrumDefense, MultiKrumDefense,
)
from flcore.security.monitor import UpdateMonitor


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.args = args
        self.device = args.device
        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.global_rounds = args.global_rounds
        self.local_epochs = args.local_epochs
        self.batch_size = args.batch_size
        self.learning_rate = args.local_learning_rate
        self.global_model = copy.deepcopy(args.model)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.random_join_ratio = args.random_join_ratio
        self.num_join_clients = int(self.num_clients * self.join_ratio)
        self.current_num_join_clients = self.num_join_clients
        self.few_shot = args.few_shot
        self.algorithm = args.algorithm
        self.time_select = args.time_select
        self.goal = args.goal
        self.time_threthold = args.time_threthold
        self.experiment_log_dir = getattr(args, 'experiment_log_dir', '.')
        self.save_folder_name = os.path.join(args.save_folder_name, f"run{times}")
        self.model_output_dir = os.path.join(self.experiment_log_dir, "models")
        self.result_output_dir = os.path.join(self.experiment_log_dir, "results")
        self.top_cnt = args.top_cnt
        self.auto_break = args.auto_break

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        # extra metrics for attacks
        self.rs_OA = []
        self.rs_SA = []
        self.rs_ASR = []

        self.times = times
        self.eval_gap = args.eval_gap
        self.client_drop_rate = args.client_drop_rate
        self.train_slow_rate = args.train_slow_rate
        self.send_slow_rate = args.send_slow_rate

        self.dlg_eval = args.dlg_eval
        self.dlg_gap = args.dlg_gap
        self.batch_num_per_client = args.batch_num_per_client

        self.num_new_clients = args.num_new_clients
        self.new_clients = []
        self.eval_new_clients = False
        self.fine_tuning_epoch_new = args.fine_tuning_epoch_new

        self.attack_type = getattr(args, 'attack_type', 'none')
        self.attack_malicious_ratio = getattr(args, 'attack_malicious_ratio', 0.0)
        self.attack_seed = getattr(args, 'attack_seed', 0)
        self.attack_exclude_malicious_in_eval = getattr(args, 'attack_exclude_malicious_in_eval', False)
        self.model_poison_attack = None
        normalized_attack = self.attack_type.lower().replace('-', '').replace('_', '')
        if normalized_attack in {'minmax', 'minsum'}:
            self.model_poison_attack = MinMaxMinSumAttack(
                attack=self.attack_type,
                deviation=getattr(args, 'attack_deviation', 'std'),
                lambda_init=getattr(args, 'attack_lambda_init', 10.0),
                tolerance=getattr(args, 'attack_search_tolerance', 1e-5),
                max_steps=getattr(args, 'attack_search_max_steps', 80),
            )
        elif normalized_attack == 'fang':
            self.model_poison_attack = FangKrumAttack(
                assumed_malicious_ratio=getattr(args, 'fang_assumed_malicious_ratio', 0.2),
                lambda_init=getattr(args, 'attack_lambda_init', 10.0),
                tolerance=getattr(args, 'attack_search_tolerance', 1e-5),
                max_steps=getattr(args, 'attack_search_max_steps', 80),
            )
        m = int(self.num_clients * self.attack_malicious_ratio)
        if m > 0:
            rs = np.random.RandomState(self.attack_seed)
            self.malicious_ids = sorted(rs.choice(range(self.num_clients), m, replace=False).tolist())
        else:
            self.malicious_ids = []

        if len(self.malicious_ids) > 0:
            try:
                self.save_item(self.malicious_ids, 'malicious_ids')
            except Exception:
                pass

        self.defense_type = getattr(args, 'defense_type', 'none').lower()
        self.defense_start_round = int(getattr(args, 'defense_start_round', 0))
        if self.defense_start_round < 0:
            raise ValueError("defense_start_round must be non-negative")
        self.dnc_defense = None
        self.cosine_cluster_defense = None
        self.robust_defense = None
        self.defense_aggregated_model = None
        if self.defense_type == 'dnc':
            self.dnc_defense = DnCDefense(
                assumed_malicious_ratio=getattr(args, 'dnc_assumed_malicious_ratio', 0.2),
                filter_fraction=getattr(args, 'dnc_filter_fraction', 1.0),
                iterations=getattr(args, 'dnc_iterations', 1),
                subsample_dim=getattr(args, 'dnc_subsample_dim', 10_000),
                seed=getattr(args, 'dnc_seed', 0),
            )
            self.robust_defense = self.dnc_defense
        elif self.defense_type == 'cosine_cluster':
            self.cosine_cluster_defense = CosineClusterDefense(
                max_iterations=getattr(args, 'cosine_cluster_max_iterations', 50),
                tolerance=getattr(args, 'cosine_cluster_tolerance', 1e-6),
            )
            self.robust_defense = self.cosine_cluster_defense
        elif self.defense_type == 'krum':
            self.robust_defense = KrumDefense(args.robust_assumed_malicious_ratio)
        elif self.defense_type == 'multikrum':
            self.robust_defense = MultiKrumDefense(
                args.robust_assumed_malicious_ratio, args.multikrum_selection_count)
        elif self.defense_type == 'trimmedmean':
            self.robust_defense = CoordinateDefense(
                'trimmedmean', args.robust_assumed_malicious_ratio)
        elif self.defense_type == 'median':
            self.robust_defense = CoordinateDefense(
                'median', args.robust_assumed_malicious_ratio)
        elif self.defense_type == 'bulyan':
            self.robust_defense = BulyanDefense(args.robust_assumed_malicious_ratio)
        elif self.defense_type != 'none':
            raise ValueError(f"Unsupported defense type: {self.defense_type}")

        self.update_monitor = None
        self.update_monitor_round = 0
        # A defense run must always leave an audit trail, even without -amon.
        if getattr(args, 'attack_monitor', False) or self.robust_defense is not None:
            monitor_root = getattr(
                args, 'attack_monitor_dir', '../results/attack_monitor')
            run_name = f"run{self.times}"
            self.update_monitor = UpdateMonitor(
                output_dir=os.path.join(monitor_root, run_name),
                l0_epsilon=getattr(args, 'attack_monitor_l0_epsilon', 0.0),
                plot_every=getattr(args, 'attack_monitor_plot_every', 10),
            )
            print(f"Attack/defense monitoring enabled: {self.update_monitor.output_dir}")


    def set_clients(self, clientObj):
        for i, train_slow, send_slow in zip(range(self.num_clients), self.train_slow_clients, self.send_slow_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=train_slow, 
                            send_slow=send_slow,
                            is_malicious=(i in getattr(self, 'malicious_ids', [])))
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        if self.random_join_ratio:
            self.current_num_join_clients = np.random.choice(range(self.num_join_clients, self.num_clients+1), 1, replace=False)[0]
        else:
            self.current_num_join_clients = self.num_join_clients
        selected_clients = list(np.random.choice(self.clients, self.current_num_join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            start_time = time.time()
            
            client.set_parameters(self.global_model)

            client.send_time_cost['num_rounds'] += 1
            client.send_time_cost['total_cost'] += 2 * (time.time() - start_time)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        active_clients = random.sample(
            self.selected_clients, int((1-self.client_drop_rate) * self.current_num_join_clients))

        self.uploaded_ids = []
        self.uploaded_weights = []
        self.uploaded_models = []
        self.defense_aggregated_model = None
        tot_samples = 0
        for client in active_clients:
            try:
                client_time_cost = client.train_time_cost['total_cost'] / client.train_time_cost['num_rounds'] + \
                        client.send_time_cost['total_cost'] / client.send_time_cost['num_rounds']
            except ZeroDivisionError:
                client_time_cost = 0
            if client_time_cost <= self.time_threthold:
                tot_samples += client.train_samples
                self.uploaded_ids.append(client.id)
                self.uploaded_weights.append(client.train_samples)
                self.uploaded_models.append(client.model)
        if self.model_poison_attack is not None:
            stats = self.model_poison_attack.apply(
                global_model=self.global_model,
                uploaded_ids=self.uploaded_ids,
                uploaded_models=self.uploaded_models,
                malicious_ids=self.malicious_ids,
            )
            if stats is not None:
                print(
                    f"[{stats.attack.upper()}] malicious={stats.malicious_clients} "
                    f"lambda={stats.lambda_value:.6g} "
                    f"score={stats.score:.6g} threshold={stats.threshold:.6g}"
                )
        
        if self.attack_type == 'free_rider' and len(getattr(self, 'malicious_ids', [])) > 0:
            for idx, cid in enumerate(self.uploaded_ids):
                if cid in self.malicious_ids:
                    model = self.uploaded_models[idx]
                    model.train()
                    for p in model.parameters():
                        try:
                            noise = torch.normal(mean=0.0, std=0.5, size=p.data.shape, device=p.data.device, dtype=p.data.dtype)
                            p.data = noise
                        except Exception:
                            noise = torch.normal(mean=0.0, std=0.5, size=p.data.shape).to(p.data.device)
                            p.data = noise

        # Preserve the actual submitted models for an auditable defense decision.
        pre_defense_ids = list(self.uploaded_ids)
        pre_defense_models = list(self.uploaded_models)
        pre_defense_weights = list(self.uploaded_weights)
        defense_result = None
        if (self.robust_defense is not None
                and self.update_monitor_round >= self.defense_start_round):
            defense_result = self.robust_defense.filter(
                global_model=self.global_model,
                uploaded_models=pre_defense_models,
                uploaded_weights=pre_defense_weights,
                round_id=self.update_monitor_round,
            )

        if self.update_monitor is not None:
            self.update_monitor.record(
                round_id=self.update_monitor_round,
                global_model=self.global_model,
                uploaded_ids=pre_defense_ids,
                uploaded_models=pre_defense_models,
                uploaded_weights=pre_defense_weights,
                malicious_ids=self.malicious_ids,
                selected_ids=[client.id for client in self.selected_clients],
                defense_result=defense_result,
            )
            self.update_monitor_round += 1

        if defense_result is not None:
            self.defense_aggregated_model = defense_result.aggregated_model
            kept = defense_result.kept_positions
            self.uploaded_ids = [pre_defense_ids[index] for index in kept]
            self.uploaded_models = [pre_defense_models[index] for index in kept]
            self.uploaded_weights = [pre_defense_weights[index] for index in kept]
            print(
                f"[{defense_result.defense}] accepted={self.uploaded_ids} "
                f"blocked={[pre_defense_ids[index] for index in defense_result.removed_positions]}"
            )

        # FedAvg weights must describe the final set that is actually aggregated.
        final_samples = sum(self.uploaded_weights)
        if final_samples <= 0:
            raise RuntimeError("No client updates remain for aggregation")
        self.uploaded_weights = [weight / final_samples for weight in self.uploaded_weights]

    def aggregate_parameters(self):
        assert (len(self.uploaded_models) > 0)

        if self.defense_aggregated_model is not None:
            self._aggregate_model_buffers(
                self.defense_aggregated_model, self.uploaded_models, self.uploaded_weights)
            self.global_model = copy.deepcopy(self.defense_aggregated_model)
            self.defense_aggregated_model = None
            return

        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for param in self.global_model.parameters():
            param.data.zero_()
            
        for w, client_model in zip(self.uploaded_weights, self.uploaded_models):
            self.add_parameters(w, client_model)
        self._aggregate_model_buffers(
            self.global_model, self.uploaded_models, self.uploaded_weights)

    def add_parameters(self, w, client_model):
        for server_param, client_param in zip(self.global_model.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    @staticmethod
    def _aggregate_model_buffers(target_model, source_models, weights):
        """Aggregate non-parameter state, notably BatchNorm running statistics.

        ``nn.Module.parameters()`` excludes buffers, so parameter-only FedAvg
        silently kept the first uploaded client's BatchNorm state.  Floating
        buffers are sample-weight averaged just like model parameters.  The
        integral BatchNorm counter is kept at the largest observed value;
        averaging it would produce an invalid fractional counter.
        """
        if not source_models:
            return
        if len(source_models) != len(weights):
            raise ValueError("Model-buffer aggregation requires one weight per source model")

        target_buffers = dict(target_model.named_buffers())
        if not target_buffers:
            return
        source_buffers = [dict(model.named_buffers()) for model in source_models]
        expected_names = set(target_buffers)
        if any(set(buffers) != expected_names for buffers in source_buffers):
            raise ValueError("Uploaded models do not share the same registered buffers")

        with torch.no_grad():
            for name, target_buffer in target_buffers.items():
                values = [buffers[name].detach() for buffers in source_buffers]
                if target_buffer.is_floating_point() or target_buffer.is_complex():
                    target_buffer.zero_()
                    for weight, value in zip(weights, values):
                        target_buffer.add_(value.to(
                            device=target_buffer.device, dtype=target_buffer.dtype), alpha=float(weight))
                elif name.endswith("num_batches_tracked"):
                    target_buffer.copy_(torch.stack([
                        value.to(device=target_buffer.device, dtype=target_buffer.dtype)
                        for value in values
                    ]).amax(dim=0))
                else:
                    target_buffer.copy_(values[0].to(
                        device=target_buffer.device, dtype=target_buffer.dtype))

    def _global_model_path(self):
        os.makedirs(self.model_output_dir, exist_ok=True)
        return os.path.join(
            self.model_output_dir,
            f"{self.dataset}_{self.algorithm}_server_run{self.times}.pt",
        )

    def save_global_model(self):
        torch.save(self.global_model, self._global_model_path())

    def load_model(self):
        model_path = self._global_model_path()
        assert os.path.exists(model_path)
        self.global_model = torch.load(model_path)

    def model_exists(self):
        return os.path.exists(self._global_model_path())
        
    def save_results(self):
        if self.update_monitor is not None:
            self.update_monitor.plot()
        algo = self.dataset + "_" + self.algorithm
        result_path = self.result_output_dir
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if (len(self.rs_test_acc)):
            algo = algo + "_" + self.goal + "_" + str(self.times)
            file_path = os.path.join(result_path, "{}.h5".format(algo))
            print("File path: " + file_path)

            with h5py.File(file_path, 'w') as hf:
                hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
                hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
                hf.create_dataset('rs_train_loss', data=self.rs_train_loss)
                # extra metrics
                hf.create_dataset('rs_OA', data=self.rs_OA)
                hf.create_dataset('rs_SA', data=self.rs_SA)
                hf.create_dataset('rs_ASR', data=self.rs_ASR)


    def save_item(self, item, item_name):
        if not os.path.exists(self.save_folder_name):
            os.makedirs(self.save_folder_name)
        torch.save(item, os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def load_item(self, item_name):
        return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    def test_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            self.fine_tuning_new_clients()
            return self.test_metrics_new_clients()
        
        num_samples = []
        tot_correct = []
        tot_auc = []
        # aggregate source-class stats for SA/ASR
        tot_src_total = []
        tot_src_correct = []
        tot_src_to_tgt = []
        for c in self.clients:
            if self.attack_exclude_malicious_in_eval and getattr(self, 'malicious_ids', None) is not None:
                if c.id in self.malicious_ids:
                    continue
            ct, ns, auc, src_total, src_correct, src_to_tgt = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
            tot_src_total.append(src_total)
            tot_src_correct.append(src_correct)
            tot_src_to_tgt.append(src_to_tgt)

        ids = [c.id for c in self.clients]

        return ids, num_samples, tot_correct, tot_auc, tot_src_total, tot_src_correct, tot_src_to_tgt

    def train_metrics(self):
        if self.eval_new_clients and self.num_new_clients > 0:
            return [0], [1], [0]
        
        num_samples = []
        losses = []
        for c in self.clients:
            if self.attack_exclude_malicious_in_eval and getattr(self, 'malicious_ids', None) is not None:
                if c.id in self.malicious_ids:
                    continue
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    def evaluate(self, acc=None, loss=None):
        stats = self.test_metrics()
        stats_train = self.train_metrics()

        test_acc = sum(stats[2])*1.0 / sum(stats[1])
        test_auc = sum(stats[3])*1.0 / sum(stats[1])
        # OA/SA/ASR
        src_total_sum = sum(stats[4]) if len(stats[4]) > 0 else 0
        src_correct_sum = sum(stats[5]) if len(stats[5]) > 0 else 0
        src_to_tgt_sum = sum(stats[6]) if len(stats[6]) > 0 else 0
        OA = test_acc
        SA = (src_correct_sum / src_total_sum) if src_total_sum > 0 else 0.0
        ASR = (src_to_tgt_sum / src_total_sum) if src_total_sum > 0 else 0.0
        train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
        accs = [a / n for a, n in zip(stats[2], stats[1])]
        aucs = [a / n for a, n in zip(stats[3], stats[1])]
        
        if acc == None:
            self.rs_test_acc.append(test_acc)
        else:
            acc.append(test_acc)
        
        if loss == None:
            self.rs_train_loss.append(train_loss)
        else:
            loss.append(train_loss)

        # record extra metrics
        self.rs_OA.append(OA)
        self.rs_SA.append(SA)
        self.rs_ASR.append(ASR)

        if self.update_monitor is not None:
            # ``update_monitor_round`` is the current communication round here:
            # evaluation happens before that round's client uploads are received.
            self.update_monitor.record_global_metrics(
                round_id=self.update_monitor_round,
                test_accuracy=test_acc,
                test_auc=test_auc,
                train_loss=train_loss,
                overall_accuracy=OA,
                source_accuracy=SA,
                attack_success_rate=ASR,
                # Untargeted poisoning has no source/target labels; its printed
                # zero ASR must not be plotted as a meaningful backdoor metric.
                asr_available=src_total_sum > 0,
            )

        print("Averaged Train Loss: {:.4f}".format(train_loss))
        print("Averaged Test Accuracy: {:.4f}".format(test_acc))
        print("Averaged Test AUC: {:.4f}".format(test_auc))
        # self.print_(test_acc, train_acc, train_loss)
        print("Std Test Accuracy: {:.4f}".format(np.std(accs)))
        print("Std Test AUC: {:.4f}".format(np.std(aucs)))
        # print OA/SA/ASR
        print("Overall Accuracy (OA): {:.4f}".format(OA))
        print("Source Accuracy (SA): {:.4f}".format(SA))
        print("Attack Success Rate (ASR): {:.4f}".format(ASR))

    def print_(self, test_acc, test_auc, train_loss):
        print("Average Test Accuracy: {:.4f}".format(test_acc))
        print("Average Test AUC: {:.4f}".format(test_auc))
        print("Average Train Loss: {:.4f}".format(train_loss))

    def check_done(self, acc_lss, top_cnt=None, div_value=None):
        for acc_ls in acc_lss:
            if top_cnt is not None and div_value is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_top and find_div:
                    pass
                else:
                    return False
            elif top_cnt is not None:
                find_top = len(acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
                if find_top:
                    pass
                else:
                    return False
            elif div_value is not None:
                find_div = len(acc_ls) > 1 and np.std(acc_ls[-top_cnt:]) < div_value
                if find_div:
                    pass
                else:
                    return False
            else:
                raise NotImplementedError
        return True

    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)

            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break

                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))

            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
                
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')

        # self.save_item(items, f'DLG_{R}')

    def set_new_clients(self, clientObj):
        for i in range(self.num_clients, self.num_clients + self.num_new_clients):
            train_data = read_client_data(self.dataset, i, is_train=True, few_shot=self.few_shot)
            test_data = read_client_data(self.dataset, i, is_train=False, few_shot=self.few_shot)
            client = clientObj(self.args, 
                            id=i, 
                            train_samples=len(train_data), 
                            test_samples=len(test_data), 
                            train_slow=False, 
                            send_slow=False)
            self.new_clients.append(client)

    # fine-tuning on new clients
    def fine_tuning_new_clients(self):
        for client in self.new_clients:
            client.set_parameters(self.global_model)
            opt = torch.optim.SGD(client.model.parameters(), lr=self.learning_rate)
            CEloss = torch.nn.CrossEntropyLoss()
            trainloader = client.load_train_data()
            client.model.train()
            for e in range(self.fine_tuning_epoch_new):
                for i, (x, y) in enumerate(trainloader):
                    if type(x) == type([]):
                        x[0] = x[0].to(client.device)
                    else:
                        x = x.to(client.device)
                    y = y.to(client.device)
                    output = client.model(x)
                    loss = CEloss(output, y)
                    opt.zero_grad()
                    loss.backward()
                    opt.step()

    # evaluating on new clients
    def test_metrics_new_clients(self):
        num_samples = []
        tot_correct = []
        tot_auc = []
        tot_src_total = []
        tot_src_correct = []
        tot_src_to_tgt = []
        for c in self.new_clients:
            ct, ns, auc, src_total, src_correct, src_to_tgt = c.test_metrics()
            tot_correct.append(ct*1.0)
            tot_auc.append(auc*ns)
            num_samples.append(ns)
            tot_src_total.append(src_total)
            tot_src_correct.append(src_correct)
            tot_src_to_tgt.append(src_to_tgt)

        ids = [c.id for c in self.new_clients]

        return ids, num_samples, tot_correct, tot_auc, tot_src_total, tot_src_correct, tot_src_to_tgt




