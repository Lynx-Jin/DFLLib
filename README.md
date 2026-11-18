<div align="center">

# 🛡️ DFLLib

**A federated learning framework for attack and defense research**

[Implementation guide](system/flcore/security/README.md) · [Integration and validation](docs/security-alignment.md) · [License](LICENSE)

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
| BadNets | `-bd badnets` |
| Blended | `-bd blended` |
| Refool-style Reflection | `-bd refool` |
| Clean-label | `-bd clean_label` |
| Distributed Backdoor Attack (DBA) | `-bd dba` |
| Neurotoxin | `-bd neurotoxin` |
| A3FL | `-bd a3fl` |
| F3BA | `-bd f3ba` |

`reflect` aliases `refool`, and `label_consistent` aliases `clean_label`. Refool, Clean-label, A3FL, and F3BA use lightweight project adaptations. Cosine-based Min-Max and the PoisonedFL Adam adapter follow the project-specific implementations documented in the guide.

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
