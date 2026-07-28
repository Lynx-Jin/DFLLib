# 🛡️ DFLLib: Detection-oriented Federated Learning Library

DFLLib is a PyTorch research framework for **Byzantine model poisoning, robust aggregation, and auditable update-level detection** in federated learning. It extends the training infrastructure of [PFLlib](https://github.com/TsingZ0/PFLlib) with an attack–defense pipeline designed for reproducible security experiments.

> 📦 Generated datasets, checkpoints, experiment logs, and monitor outputs are excluded from version control. Generate or place datasets locally before running experiments.

## ✨ DFLLib extensions

- 🧪 **Threat-model-aware attacks:** AGR-agnostic Min-Max and Min-Sum; oracle, Krum-targeted Fang attack; label flipping; and a Gaussian free-rider baseline.
- 🧱 **Robust aggregation and detection:** Krum, Multi-Krum, Bulyan, coordinate-wise Trimmed Mean/Median, DnC random-subspace SVD filtering, and AFA-style cosine-score clustering.
- 🔎 **Per-round auditability:** records scheduled, uploaded, accepted, and blocked clients; pre-defense L0/L2/L∞ and cosine statistics; defense scores; OA, SA, ASR, and dashboard plots.
- ⚙️ **Experiment control:** independent attack and defense budgets, fixed malicious-client sampling, defense warm-up, and repeated DnC random-subspace filtering.
- 🧠 **Training/model support:** FedAvg and one-minibatch FedSGD with server SGD/Adam; FedAvgMLP, CNN, VGG11, AlexNet, and small-image ResNet18.

## 🗂️ Layout

| Path | Purpose |
|---|---|
| `system/main.py` | Experiment entry point and CLI arguments. |
| `system/flcore/security/attack/` | Poisoning attacks. |
| `system/flcore/security/defense/` | Robust aggregation and filtering. |
| `system/flcore/security/monitor/` | CSV audit trail and monitoring dashboard. |
| `system/flcore/security/README.md` | Security-module implementation notes and detailed usage. |

## 🧪 Attacks

Shared arguments: `-atk` (default `none`), `-amr` malicious-client ratio (default `0.0`), and `-aseed` malicious-ID seed (default `0`). Enable auditing with `-amon`; configure it with `-amondir attack_monitor`, `-amonplot 10`, and `-amoneps 0.0`.

| `-atk` | Method | Implementation | Method-specific arguments |
|---|---|---|---|
| `none` | Clean baseline | Leaves uploads unchanged. | None; use `-atk none -amr 0`. |
| `label_flip` | MNIST label-flip attack | For malicious clients on datasets whose names start with `MNIST`, changes training label `7 → 1`; test labels remain clean. | No source/target CLI mapping. |
| `free_rider` | Gaussian upload baseline | Replaces the complete malicious upload with independent Gaussian noise. | Noise standard deviation is currently fixed at `0.5`. |
| `minmax` | AGR-agnostic Min-Max | Uses only active malicious clients' clean local updates as proxies; searches a shared update constrained by maximum squared L2 distance. Every active malicious client submits the same update. | `-adev {unit_vec,sign,std}` (default `std`), `-alinit 10.0`, `-atol 1e-5`, `-amaxsteps 80`. |
| `minsum` | AGR-agnostic Min-Sum | Same information model and colluding update as Min-Max; constrains the sum of squared L2 distances to malicious proxies. | `-adev {unit_vec,sign,std}` (default `std`), `-alinit 10.0`, `-atol 1e-5`, `-amaxsteps 80`. |
| `fang` | Oracle Fang–Krum | Observes current benign uploads and searches an inverse-sign malicious update selected by Krum. This is a known-AGR, known-benign-update attack. | `-fmar 0.20`, `-alinit 10.0`, `-atol 1e-5`, `-amaxsteps 80`; interpret with `-def krum`. |

`-adev`: `unit_vec` uses the unit proxy-mean direction; `sign` uses coordinate-wise proxy-mean signs; `std` uses coordinate-wise proxy-update standard deviations.

## 🧱 Defenses

All defenses use `-def` (default `none`) and `-defstart` (default `0`). A defense is applied only when `round >= defstart`. `-rmar 0.20` is the assumed Byzantine ratio for Krum, Multi-Krum, Bulyan, Trimmed Mean, and Median; it is distinct from the attack truth `-amr`.

| `-def` | Method | Implementation | Arguments and constraints |
|---|---|---|---|
| `none` | Standard aggregation | Sample-weighted FedAvg; FedSGD applies the server optimizer to the mean gradient. | None. |
| `krum` | Krum | Selects the update with the lowest sum of squared distances to its `n-f-2` nearest neighbors. | `-rmar 0.20`; requires `n ≥ 2f+3`. |
| `multikrum` | Multi-Krum | Selects several Krum winners and averages them. | `-rmar 0.20`; `-mks 0` selects the maximal legal count `n-2f-3`, or provide a positive count; requires `n ≥ 2f+3`. |
| `bulyan` | Bulyan | Multi-Krum candidate selection followed by coordinate-wise median-centered averaging. | `-rmar 0.20`; requires `n ≥ 4f+3`. |
| `trimmedmean` | Trimmed Mean | Removes the largest and smallest `f` values per coordinate, then averages. | `-rmar 0.20`; requires `n > 2f`; coordinate-level aggregation. |
| `median` | Coordinate Median | Takes the median per coordinate. | `-rmar 0.20`; coordinate-level aggregation. |
| `cosine_cluster` | AFA-style cosine clustering | Clusters `cos(u_i, Σ_j w_j u_j)` with deterministic 1-D two-means, then retains one cluster. This is a baseline, not an exact AFA implementation. | `-ccselect {high_mean,largest_cluster}` (default `high_mean`); `high_mean` retains the cluster with higher mean cosine, `largest_cluster` retains the larger client cluster and uses mean cosine for ties; `-cciter 50`; `-cctol 1e-6`. |
| `dnc` | DnC spectral filtering | Samples parameter coordinates, centers updates, performs SVD, and filters high squared projections on the first right singular vector. Repeated runs intersect survivor sets. | `-dmar 0.20`, `-df 1.0`, `-di 1`, `-db 10000`, `-dseed 0`. |

## 🚀 Examples

Run from `system/`.

```bash
# Clean FedSGD baseline
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk none -amr 0 -amon

# AGR-agnostic Min-Max with cosine clustering
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk minmax -amr 0.20 -adev sign \
  -def cosine_cluster -ccselect high_mean -cciter 50 -cctol 1e-6 -amon

# Min-Sum with majority-cluster selection and defense warm-up
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk minsum -amr 0.20 -adev sign \
  -def cosine_cluster -ccselect largest_cluster -defstart 50 -amon

# DnC oracle-budget ablation (dmar = amr)
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk minmax -amr 0.20 -adev sign \
  -def dnc -dmar 0.20 -df 1.0 -di 1 -db 10000 -dseed 0 -amon
```

## 📊 Audit outputs

| File | Contents |
|---|---|
| `round_audit.csv` | Scheduled/uploaded/accepted/blocked IDs, benign/malicious counts, defense scores, aggregation mode, and cosine-cluster selection rule. |
| `client_metrics.csv` | Pre-defense L0/L2/L∞, cosine to the weighted mean update, defense score, and decision for every upload. |
| `group_summary.csv` | Offline benign/malicious group statistics. |
| `global_metrics.csv` | Test accuracy/AUC/loss, OA, SA, ASR, and ASR availability. |
| `monitor_dashboard.png` | Per-round detection and task-metric plots. |

For untargeted update attacks such as Min-Max and Min-Sum, ASR is not a meaningful security metric. Report clean/attacked accuracy, malicious acceptance and blocking rates, benign false-blocking rate, and the actual number of uploads per round. If `-amr`, `-rmar`, or `-dmar` are intentionally matched, report the setting as an oracle-budget ablation.

## 📚 Attribution

DFLLib is built on the training framework and selected infrastructure of [PFLlib](https://github.com/TsingZ0/PFLlib). Please comply with the upstream Apache-2.0 license and provide appropriate attribution when using this repository.
