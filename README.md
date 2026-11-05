# DFLlib

A detection-oriented federated learning library built on [PFLlib](https://github.com/TsingZ0/PFLlib).

## Attacks

| Attack | Original paper / source | Venue |
|---|---|---|
| PoisonedFL | [Model Poisoning Attacks to Federated Learning via Multi-Round Consistency][poisonedfl] | **CVPR 2025** |
| Min-Max | [Manipulating the Byzantine: Optimizing Model Poisoning Attacks and Defenses for Federated Learning][minmax] | **NDSS 2021** |
| Min-Sum | [Same paper as Min-Max][minmax] | **NDSS 2021** |
| Cosine-Tailored Min-Max | Project adaptation of [Min-Max][minmax] | — |
| Fang (Krum-targeted) | [Local Model Poisoning Attacks to Byzantine-Robust Federated Learning][fang] | **USENIX Security 2020** |
| Label Flipping | Baseline | — |
| Gaussian Free-Rider | Baseline | — |

The cosine-tailored Min-Max and Adam-proxy PoisonedFL variants are project adaptations of the cited methods.

## Defenses

- Krum
- Multi-Krum
- Bulyan
- Trimmed Mean
- Coordinate-wise Median
- Divide-and-Conquer (DnC)
- Cosine Clustering (AFA-style baseline)
- KMeans (majority-cluster selection)
- KMeans + DBSCAN (with historical consistency)

[poisonedfl]: https://openaccess.thecvf.com/content/CVPR2025/html/Xie_Model_Poisoning_Attacks_to_Federated_Learning_via_Multi-Round_Consistency_CVPR_2025_paper.html
[minmax]: https://www.ndss-symposium.org/wp-content/uploads/ndss2021_6C-3_24498_paper.pdf
[fang]: https://www.usenix.org/conference/usenixsecurity20/presentation/fang
