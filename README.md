<div align="center">

# 🛡️ DFLLib

**A federated learning framework for attack and defense research**

[Implementation guide](system/flcore/security/README.md) · [Data-free distillation](docs/single-teacher-datafree.md) · [Federated ZSKT](docs/federated-zskt.md) · [License](LICENSE)

</div>

Built on [PFLlib](https://github.com/TsingZ0/PFLlib), with integrated poisoning attacks, image backdoors, robust aggregation, and model repair. Security experiments support **FedAvg** and **FedSGD** with server-side SGD or Adam.

## 🧪 Poisoning attacks

| Method | Option |
|---|---|
| Label Flipping | `-atk label_flip` |
| Gaussian Free-Rider | `-atk free_rider` |
| Min-Max | `-atk minmax` |
| Min-Sum | `-atk minsum` |
| Cosine-constrained Min-Max | `-atk minmax -amv cosine` |
| Cosine-feasibility Min-Max | `-atk minmax -amv cosine_oracle` |
| Fang-Krum | `-atk fang` |
| PoisonedFL | `-atk poisonedfl` |

## 🎯 Image backdoors

| Method | Option |
|---|---|
| BadNets · reference / legacy presets | `-bd badnets --backdoor_preset reference` |
| Blended | `-bd blended` |
| Refool-style Reflection | `-bd refool` |
| Clean-label | `-bd clean_label` |
| Distributed Backdoor Attack (DBA) | `-bd dba` |
| Neurotoxin | `-bd neurotoxin` |
| A3FL | `-bd a3fl` |
| F3BA | `-bd f3ba` |

`reflect` aliases `refool`, and `label_consistent` aliases `clean_label`. Refool, Clean-label, A3FL, and F3BA use lightweight project adaptations. Cosine-based Min-Max and the PoisonedFL Adam adapter follow the project-specific implementations documented in the guide.

BadNets defaults to the **reference** preset: poison counts use 20% of all local records, selected from non-target examples with fixed per-client indices; the trigger is a 4×4 white square at the lower-right with a one-pixel margin. `--backdoor_preset legacy` retains the previous sampling and trigger defaults.

## 🔁 Data-free distillation

| Protocol | Entry point | Guide |
|---|---|---|
| Single teacher → same-size CNN student | `bash tools/reproduce_single_teacher_datafree.sh badnets` | [Single-teacher setup](docs/single-teacher-datafree.md) |
| FedAvg → multi-teacher ZSKT | `bash tools/reproduce_federated_zskt.sh badnets` | [Federated setup](docs/federated-zskt.md) |
| Dirichlet α=0.1 client splits | `FL_ZSKT_PARTITION=dirichlet FL_ZSKT_ALPHA=0.1` | [Non-IID comparison](docs/federated-zskt.md#dirichlet-α01-对照) |

Run from `system/` after preparing the required CIFAR-10 data. The federated launcher also provides a `fedavg` control with the same BadNets and local training settings, and a `clean` mode without malicious clients. Server distillation uses generator outputs; the held-out data is used only for evaluation. These are research protocols, and their effect on clean accuracy and ASR must be measured.

## 🧱 Defenses and model repair

| Method | Option or tool |
|---|---|
| Krum | `-def krum` |
| Multi-Krum | `-def multikrum` |
| Bulyan | `-def bulyan` |
| Trimmed Mean | `-def trimmedmean` |
| Coordinate-wise Median | `-def median` |
| Divide-and-Conquer (DnC) | `-def dnc` |
| AFA-style Cosine Clustering | `-def cosine_cluster` |
| Scalar KMeans | `-def scalar_kmeans` |
| Scalar KMeans + DBSCAN | `-def scalar_dbscan` |
| Clean-reference Prototype Screening | `-def proto-clean` |
| Prototype Probes · diagnostic scoring | `-def proto-probe` |
| PGBD · post-training model repair | `python -m tools.run_pgbd_repair` |

Prototype Probes score all clients and retain every upload. PGBD uses labeled clean reference samples. Detailed assumptions, parameters, and evaluation tools are covered in the [implementation guide](system/flcore/security/README.md).

## 📚 Attribution

DFLLib builds on [PFLlib](https://github.com/TsingZ0/PFLlib) under the Apache-2.0 license. Please retain upstream attribution when using or extending the framework.
