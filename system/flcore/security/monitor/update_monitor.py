"""Audit and visualize client uploads before robust aggregation."""

import csv
import json
import os

import torch


class UpdateMonitor:
    """Record label-free update statistics, defense decisions and evaluation metrics.

    Client-level update statistics always describe the *pre-defense* uploads.
    The cosine statistic is intentionally computed on model updates, rather than
    on full model parameters: each update is compared with the sample-weighted
    mean update of all uploads in the same round.  This is the AFA-style cosine
    reference and does not need the (audit-only) malicious-client labels.
    """

    METRICS = ("l0", "l2", "linf", "cosine_to_weighted_mean_update")

    def __init__(self, output_dir, l0_epsilon=0.0, plot_every=10):
        if l0_epsilon < 0:
            raise ValueError("l0_epsilon must be non-negative")
        if plot_every <= 0:
            raise ValueError("plot_every must be positive")

        self.output_dir = output_dir
        self.l0_epsilon = float(l0_epsilon)
        self.plot_every = int(plot_every)
        self.client_csv = os.path.join(output_dir, "client_metrics.csv")
        self.summary_csv = os.path.join(output_dir, "group_summary.csv")
        self.audit_csv = os.path.join(output_dir, "round_audit.csv")
        self.global_csv = os.path.join(output_dir, "global_metrics.csv")
        self.plot_path = os.path.join(output_dir, "monitor_dashboard.png")
        self.summary_rows = []
        self.client_rows = []
        self.audit_rows = []
        self.global_rows = []
        os.makedirs(output_dir, exist_ok=True)

        self._write_header(
            self.client_csv,
            ["round", "client_id", "group", "is_malicious"] + list(self.METRICS) + [
                "defense", "defense_score", "defense_decision"],
        )
        self._write_header(
            self.summary_csv,
            ["round", "group", "count"] + [
                f"{metric}_{suffix}"
                for metric in self.METRICS
                for suffix in ("mean", "std")
            ],
        )
        self._write_header(
            self.audit_csv,
            ["round", "defense", "scheduled_client_ids", "uploaded_client_ids",
             "global_malicious_ids", "uploaded_malicious_ids", "accepted_client_ids",
             "blocked_client_ids", "accepted_malicious_ids", "blocked_malicious_ids",
             "uploaded_count", "uploaded_malicious_count", "uploaded_benign_count",
             "accepted_count", "accepted_malicious_count", "accepted_benign_count",
             "blocked_count", "blocked_malicious_count", "blocked_benign_count",
             "aggregation_mode", "selection_rule", "score_by_client",
             "sampled_dimension_count"],
        )
        self._write_header(
            self.global_csv,
            ["round", "test_accuracy", "test_auc", "train_loss",
             "overall_accuracy", "source_accuracy", "attack_success_rate",
             "asr_available"],
        )

    @staticmethod
    def _write_header(path, fields):
        with open(path, "w", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=fields).writeheader()

    @staticmethod
    def _flatten_model(model):
        parts = [parameter.detach().reshape(-1) for parameter in model.parameters()]
        if not parts:
            raise ValueError("Cannot monitor a model with no trainable parameters")
        return torch.cat(parts).to(dtype=torch.float32, device="cpu")

    @staticmethod
    def _mean_std(values):
        tensor = torch.tensor(values, dtype=torch.float64)
        return float(tensor.mean()), float(tensor.std(unbiased=False))

    @staticmethod
    def _cosine(left, right):
        denominator = torch.norm(left) * torch.norm(right)
        if denominator <= torch.finfo(left.dtype).eps:
            return 0.0
        return float(torch.dot(left, right) / denominator)

    def _client_row(self, round_id, client_id, update, weighted_mean_update,
                    malicious_ids, defense="none", defense_score=None,
                    defense_decision="accepted"):
        is_malicious = client_id in malicious_ids
        return {
            "round": int(round_id),
            "client_id": int(client_id),
            "group": "malicious" if is_malicious else "benign",
            "is_malicious": int(is_malicious),
            "l0": int(torch.count_nonzero(torch.abs(update) > self.l0_epsilon)),
            "l2": float(torch.norm(update, p=2)),
            "linf": float(torch.norm(update, p=float("inf"))),
            "cosine_to_weighted_mean_update": self._cosine(update, weighted_mean_update),
            "defense": defense,
            "defense_score": "" if defense_score is None else float(defense_score),
            "defense_decision": defense_decision,
        }

    def record_global_metrics(self, round_id, test_accuracy, test_auc, train_loss,
                              overall_accuracy, source_accuracy,
                              attack_success_rate, asr_available):
        """Persist server-side evaluation values for the same dashboard.

        ``asr_available`` distinguishes a real targeted/backdoor ASR from the
        zero placeholder produced by an untargeted experiment with no source and
        target labels configured.
        """
        row = {
            "round": int(round_id),
            "test_accuracy": float(test_accuracy),
            "test_auc": float(test_auc),
            "train_loss": float(train_loss),
            "overall_accuracy": float(overall_accuracy),
            "source_accuracy": float(source_accuracy),
            "attack_success_rate": float(attack_success_rate),
            "asr_available": int(bool(asr_available)),
        }
        with open(self.global_csv, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=row.keys()).writerow(row)
        self.global_rows.append(row)

    def record(self, round_id, global_model, uploaded_ids,
               uploaded_models, uploaded_weights, malicious_ids, selected_ids=None,
               defense_result=None):
        """Record uploads and the defense decision before aggregation."""
        if not uploaded_ids:
            return

        malicious_ids = set(malicious_ids)
        global_vector = self._flatten_model(global_model)
        if len(uploaded_weights) != len(uploaded_ids):
            raise ValueError("Monitor requires one pre-defense weight per uploaded model")
        updates = [self._flatten_model(model) - global_vector for model in uploaded_models]
        weights = torch.tensor(uploaded_weights, dtype=torch.float32)
        weight_sum = weights.sum()
        if weight_sum <= 0:
            raise ValueError("Monitor requires positive total pre-defense upload weight")
        weighted_mean_update = (
            torch.stack(updates) * (weights / weight_sum).unsqueeze(1)
        ).sum(dim=0)
        selected_ids = [] if selected_ids is None else list(selected_ids)
        scores = {}
        kept_positions = set(range(len(uploaded_ids)))
        defense = "none"
        sampled_dimension_count = 0
        aggregation_mode = "selection"
        selection_rule = ""
        if defense_result is not None:
            defense = defense_result.defense
            scores = dict(zip(uploaded_ids, defense_result.scores))
            kept_positions = set(defense_result.kept_positions)
            sampled_dimension_count = len(defense_result.sampled_dimensions)
            aggregation_mode = getattr(defense_result, "aggregation_mode", "selection")
            selection_rule = getattr(defense_result, "selection_rule", "")

        rows = [
            self._client_row(
                round_id, client_id, update, weighted_mean_update, malicious_ids,
                defense=defense,
                defense_score=scores.get(client_id),
                defense_decision=("coordinate_aggregated" if aggregation_mode == "coordinate"
                                  else "accepted" if position in kept_positions else "blocked"),
            )
            for position, (client_id, update) in enumerate(zip(uploaded_ids, updates))
        ]
        with open(self.client_csv, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=rows[0].keys()).writerows(rows)
        self.client_rows.extend(rows)

        summaries = []
        for group in ("benign", "malicious"):
            group_rows = [row for row in rows if row["group"] == group]
            if not group_rows:
                continue
            summary = {"round": int(round_id), "group": group, "count": len(group_rows)}
            for metric in self.METRICS:
                mean, std = self._mean_std([row[metric] for row in group_rows])
                summary[f"{metric}_mean"] = mean
                summary[f"{metric}_std"] = std
            summaries.append(summary)
        with open(self.summary_csv, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=summaries[0].keys()).writerows(summaries)
        self.summary_rows.extend(summaries)

        accepted_ids = [client_id for position, client_id in enumerate(uploaded_ids)
                        if position in kept_positions]
        blocked_ids = [client_id for position, client_id in enumerate(uploaded_ids)
                       if position not in kept_positions]
        accepted_malicious_ids = [client_id for client_id in accepted_ids if client_id in malicious_ids]
        blocked_malicious_ids = [client_id for client_id in blocked_ids if client_id in malicious_ids]
        uploaded_malicious_ids = [client_id for client_id in uploaded_ids if client_id in malicious_ids]
        audit = {
            "round": int(round_id),
            "defense": defense,
            "scheduled_client_ids": json.dumps(selected_ids),
            "uploaded_client_ids": json.dumps(list(uploaded_ids)),
            "global_malicious_ids": json.dumps(sorted(malicious_ids)),
            "uploaded_malicious_ids": json.dumps(uploaded_malicious_ids),
            "accepted_client_ids": json.dumps(accepted_ids),
            "blocked_client_ids": json.dumps(blocked_ids),
            "accepted_malicious_ids": json.dumps(accepted_malicious_ids),
            "blocked_malicious_ids": json.dumps(blocked_malicious_ids),
            "uploaded_count": len(uploaded_ids),
            "uploaded_malicious_count": len(uploaded_malicious_ids),
            "uploaded_benign_count": len(uploaded_ids) - len(uploaded_malicious_ids),
            "accepted_count": len(accepted_ids),
            "accepted_malicious_count": len(accepted_malicious_ids),
            "accepted_benign_count": len(accepted_ids) - len(accepted_malicious_ids),
            "blocked_count": len(blocked_ids),
            "blocked_malicious_count": len(blocked_malicious_ids),
            "blocked_benign_count": len(blocked_ids) - len(blocked_malicious_ids),
            "aggregation_mode": aggregation_mode,
            "selection_rule": selection_rule,
            "score_by_client": json.dumps({str(key): value for key, value in scores.items()}),
            "sampled_dimension_count": sampled_dimension_count,
        }
        with open(self.audit_csv, "a", newline="", encoding="utf-8") as handle:
            csv.DictWriter(handle, fieldnames=audit.keys()).writerow(audit)
        self.audit_rows.append(audit)

        if (round_id + 1) % self.plot_every == 0:
            self.plot()

    @staticmethod
    def _plot_group_series(axis, rows, metric, colors):
        for group in ("benign", "malicious"):
            group_rows = [row for row in rows if row["group"] == group]
            if not group_rows:
                continue
            rounds = [row["round"] for row in group_rows]
            means = [row[f"{metric}_mean"] for row in group_rows]
            stds = [row[f"{metric}_std"] for row in group_rows]
            axis.plot(rounds, means, label=group, color=colors[group])
            axis.fill_between(rounds, [mean - std for mean, std in zip(means, stds)],
                              [mean + std for mean, std in zip(means, stds)],
                              color=colors[group], alpha=0.15)

    def plot(self):
        """Write one dashboard containing accuracy, ASR, audit and update signals."""
        if not self.summary_rows and not self.global_rows:
            return
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        colors = {"benign": "#2b6cb0", "malicious": "#c53030"}
        metric_labels = {
            "l0": "L0 norm of update",
            "l2": "L2 norm of update",
            "linf": "L-infinity norm of update",
            "cosine_to_weighted_mean_update": "Cosine(update, weighted mean update)",
        }
        fig, axes = plt.subplots(3, 3, figsize=(19, 14), constrained_layout=True)
        flat_axes = axes.reshape(-1)

        accuracy_axis = flat_axes[0]
        if self.global_rows:
            rounds = [row["round"] for row in self.global_rows]
            accuracy_axis.plot(rounds, [row["test_accuracy"] for row in self.global_rows],
                               color="#2f855a", label="test accuracy")
            accuracy_axis.plot(rounds, [row["overall_accuracy"] for row in self.global_rows],
                               color="#68d391", linestyle="--", label="overall accuracy")
            if any(row["asr_available"] for row in self.global_rows):
                accuracy_axis.plot(rounds, [row["source_accuracy"] for row in self.global_rows],
                                   color="#805ad5", label="source accuracy")
                accuracy_axis.plot(rounds, [row["attack_success_rate"] for row in self.global_rows],
                                   color="#dd6b20", label="ASR")
            else:
                accuracy_axis.text(0.5, 0.08, "ASR unavailable: this run has no targeted/backdoor labels",
                                   transform=accuracy_axis.transAxes, ha="center", fontsize=8)
            accuracy_axis.legend()
        accuracy_axis.set_title("Global accuracy / targeted metrics")

        loss_axis = flat_axes[1]
        if self.global_rows:
            loss_axis.plot([row["round"] for row in self.global_rows],
                           [row["train_loss"] for row in self.global_rows], color="#3182ce")
        loss_axis.set_title("Global train loss")

        defense_axis = flat_axes[2]
        if self.audit_rows:
            rounds = [row["round"] for row in self.audit_rows]
            defense_axis.plot(rounds, [row["blocked_malicious_count"] for row in self.audit_rows],
                              color="#c53030", label="blocked malicious")
            defense_axis.plot(rounds, [row["blocked_benign_count"] for row in self.audit_rows],
                              color="#2b6cb0", label="blocked benign")
            defense_axis.plot(rounds, [row["accepted_malicious_count"] for row in self.audit_rows],
                              color="#dd6b20", linestyle="--", label="accepted malicious")
            defense_axis.legend()
        defense_axis.set_title("Defense decisions (audit labels)")

        for axis, metric in zip(flat_axes[3:7], self.METRICS):
            self._plot_group_series(axis, self.summary_rows, metric, colors)
            axis.set_title(metric_labels[metric])
            axis.legend()

        score_axis = flat_axes[7]
        for group in ("benign", "malicious"):
            rows = [row for row in self.client_rows
                    if row["group"] == group and row["defense_score"] != ""]
            if rows:
                by_round = {}
                for row in rows:
                    by_round.setdefault(row["round"], []).append(float(row["defense_score"]))
                score_axis.plot(list(by_round), [sum(values) / len(values) for values in by_round.values()],
                                color=colors[group], label=group)
        score_axis.set_title("Defense score (pre-aggregation uploads)")
        if score_axis.lines:
            score_axis.legend()

        selection_axis = flat_axes[8]
        if self.audit_rows:
            rounds = [row["round"] for row in self.audit_rows]
            selection_axis.plot(rounds, [row["accepted_benign_count"] for row in self.audit_rows],
                                color="#2b6cb0", label="accepted benign")
            selection_axis.plot(rounds, [row["accepted_malicious_count"] for row in self.audit_rows],
                                color="#c53030", label="accepted malicious")
            selection_axis.plot(rounds, [row["blocked_benign_count"] for row in self.audit_rows],
                                color="#63b3ed", linestyle="--", label="blocked benign")
            selection_axis.legend()
        selection_axis.set_title("Accepted vs blocked client counts")

        for axis in flat_axes:
            axis.set_xlabel("Communication round")
            axis.grid(alpha=0.25)
        fig.suptitle("Federated poisoning / defense monitoring dashboard", fontsize=16)
        fig.savefig(self.plot_path, dpi=180)
        plt.close(fig)
