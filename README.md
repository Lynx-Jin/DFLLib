# DFLLib: Detection-oriented Federated Learning Library

DFLLib 是一个面向**联邦学习模型投毒、鲁棒聚合与可审计检测**的 PyTorch 研究框架。项目以 [TsingZ0/PFLlib](https://github.com/TsingZ0/PFLlib) 的联邦学习训练基础设施为起点，重点扩展 Byzantine model poisoning 的攻击复现、服务端防御和逐轮实验审计能力。

它的目标不是把“准确率下降”当成唯一结论，而是让每一轮攻击和防御决策都可复查：哪些客户端被选中、哪些为实验设定的恶意客户端、实际上传了哪些更新、最终保留/屏蔽了谁，以及更新范数、余弦分数和任务指标如何变化。

> 生成的数据集、模型检查点、实验 logger 和监控结果均被 `.gitignore` 排除；克隆后请在本地生成或放置数据集。

## DFLLib 的主要贡献

- **攻击威胁模型分层。** 同时提供 AGR-agnostic 的 Min-Max / Min-Sum，以及知道良性更新和 Krum 配置的 oracle Fang-Krum；文档明确区分两者，避免把 oracle 结果误称为 agnostic 攻击。
- **鲁棒聚合与检测基准。** 集成 Krum、Multi-Krum、Bulyan、坐标级 Trimmed Mean/Median、DnC 随机子空间 SVD，以及可切换簇选择准则的 AFA-style cosine clustering。
- **可审计的攻防流水线。** 防御在服务端聚合前运行；`round_audit.csv`、`client_metrics.csv`、`group_summary.csv` 和监控总图记录攻击前上传、保留/屏蔽决策、L0/L2/L∞、与加权均值更新的余弦相似度、OA/SA/ASR 等指标。
- **适用于更新级实验的训练路径。** 支持 FedAvg 和每个选中客户端仅上传一个 minibatch 梯度的 FedSGD；FedSGD 可配服务器 SGD/Adam。模型覆盖 FedAvgMLP、CNN、VGG11、AlexNet、针对小图像适配的 ResNet18。
- **工程化实验控制。** 支持恶意客户端比例与固定随机 ID、攻击/防御预算分离、`-defstart` 防御 warm-up，以及 DnC 多次随机子空间筛选。

## 目录导览

| 路径 | 内容 |
|---|---|
| `system/main.py` | 实验入口、模型/算法/攻击/防御命令行参数。 |
| `system/flcore/security/attack/` | Min-Max、Min-Sum、Fang-Krum 等攻击实现。 |
| `system/flcore/security/defense/` | DnC、cosine cluster 和鲁棒 AGR 实现。 |
| `system/flcore/security/monitor/` | 逐轮 CSV 审计与 dashboard 绘图。 |
| `system/flcore/security/README.md` | 安全模块的实现细节、假设与完整命令说明。 |
| `dataset/` | 数据划分/生成脚本；生成后的数据不纳入版本控制。 |

## 攻击方法与参数

所有攻击共用：`-atk`（默认 `none`）、`-amr` 恶意客户端占全体客户端比例（默认 `0.0`）、`-aseed` 固定恶意客户端 ID 的随机种子（默认 `0`）。`-amon` 开启审计；`-amondir`（默认 `attack_monitor`）、`-amonplot`（默认 `10`）和 `-amoneps`（默认 `0.0`）控制输出位置、绘图频率和 L0 阈值。

| `-atk` 取值 | DFLLib 中的逻辑 | 方法专用参数（默认值与可选项） |
|---|---|---|
| `none` | 不修改客户端上传，作为 clean 对照。 | 无。建议显式写 `-atk none -amr 0`。 |
| `label_flip` | 对数据集名以 `MNIST` 开头的恶意客户端训练标签执行固定 `7 → 1` 翻转；测试标签不变。 | 无额外参数；当前不提供 source/target 映射选项。 |
| `free_rider` | 本地训练后用独立高斯噪声覆盖恶意客户端的整份上传模型。 | 无额外 CLI 参数；噪声标准差当前固定为 `0.5`，属于显式异常基线。 |
| `minmax` | 使用**本轮活跃恶意客户端自身的干净更新**作代理，构造一份共享恶意更新；该更新到任一代理的最大平方 L2 距离受约束。所有活跃恶意客户端上传完全相同的更新。 | `-adev {unit_vec,sign,std}`（`std`）；`-alinit 10.0`；`-atol 1e-5`；`-amaxsteps 80`。 |
| `minsum` | 与 Min-Max 相同的 agnostic 信息集和共享恶意更新，但约束是到全部代理的平方 L2 距离和。 | 同 Min-Max：`-adev {unit_vec,sign,std}`、`-alinit`、`-atol`、`-amaxsteps`。 |
| `fang` | Fang-style、面向 Krum 的 oracle 攻击：读取当轮良性上传，用反符号方向搜索仍会被 Krum 选中的共享恶意更新。 | `-fmar 0.20` 为攻击者已知的 Krum Byzantine 预算；另有 `-alinit 10.0`、`-atol 1e-5`、`-amaxsteps 80`。仅应与 `-def krum` 一起解释。 |

`-adev` 的含义：`unit_vec` 为代理均值的单位方向；`sign` 为代理均值的逐坐标符号；`std` 为代理更新的逐坐标标准差。它们是攻击偏移方向，而不是防御阈值。Min-Max/Min-Sum 不读取良性客户端更新，也不读取服务器所选 AGR；`fang` 恰好相反，属于已知防御的受控攻击。

## 防御方法与参数

所有防御共用 `-def`（默认 `none`）和 `-defstart`（默认 `0`）。只有通信轮 `round >= defstart` 才会实际防御；此前仍照常记录审计数据。`-rmar`（默认 `0.20`）是 Krum、Multi-Krum、Bulyan、Trimmed Mean、Median 的**公开假定 Byzantine 比例**，不等于攻击真值 `-amr`。

| `-def` 取值 | DFLLib 中的逻辑 | 参数、可选项与约束 |
|---|---|---|
| `none` | 样本数加权 FedAvg；FedSGD 则由服务器优化器更新平均梯度。 | 无。 |
| `krum` | 按最近 `n-f-2` 个邻居的平方距离和评分，仅聚合评分最低的一个更新。 | `-rmar 0.20`；要求本轮上传数 `n ≥ 2f+3`。 |
| `multikrum` | 以 Krum 分数选多个更新后平均。 | `-rmar 0.20`；`-mks 0` 表示自动取最大合法数量 `n-2f-3`，也可指定正整数；要求 `n ≥ 2f+3`。 |
| `bulyan` | 先以 Multi-Krum 选候选，再逐坐标保留接近候选中位数的值并平均。 | `-rmar 0.20`；要求 `n ≥ 4f+3`，因此 `n=100, f=25` 不合法。 |
| `trimmedmean` | 每个参数坐标独立排序，去掉两端各 `f` 个值后平均。 | `-rmar 0.20`；要求 `n > 2f`。这是坐标级而非客户端整体屏蔽。 |
| `median` | 每个参数坐标独立取中位数。 | `-rmar 0.20` 用于统一记录防御预算；同样是坐标级聚合。 |
| `cosine_cluster` | 计算 `cos(u_i, Σ_j w_j u_j)`；对一维余弦分数做确定性二均值聚类，再保留一个簇。它是 AFA-style 基线，**不是** AFA 的完整复现。 | `-ccselect {high_mean,largest_cluster}`（默认 `high_mean`）；`-cciter 50`；`-cctol 1e-6`。 |
| `dnc` | 对随机抽取的参数坐标子空间中心化并做 SVD；按首右奇异向量投影平方排除高分更新。多次子空间运行时保留集取交集。 | `-dmar 0.20`、`-df 1.0`、`-di 1`、`-db 10000`、`-dseed 0`。 |

### Cosine-cluster 的保留规则

`-ccselect` 控制二均值分簇后哪一簇被视为良性簇：

- `high_mean`（默认，保持旧行为）：保留**平均余弦相似度更高**的簇。
- `largest_cluster`：保留**客户端数量更多**的簇；若两簇大小相同，则以平均余弦相似度更高者打破平局。

两种规则使用相同的余弦分数和相同的二均值划分，区别只在最后的簇选择假设。前者假定良性更新更接近全体加权均值，后者假定良性客户端是多数；在攻击者占比接近一半、非 IID 导致良性更新多模态，或攻击者集体主导均值时，这些假设都可能失效，应以审计中的恶意接受率与良性误屏蔽率报告结果。

## 常用实验命令

以下命令从 `system/` 目录执行；其中参数仅展示接口用法，不构成通用最优超参。

```bash
# Clean FedSGD 对照
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk none -amr 0 -amon

# AGR-agnostic Min-Max + cosine 分簇：按余弦均值高的簇保留
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk minmax -amr 0.20 -adev sign \
  -def cosine_cluster -ccselect high_mean -cciter 50 -cctol 1e-6 -amon

# 同一分簇，但按簇大小保留；第 50 轮才开始防御
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk minsum -amr 0.20 -adev sign \
  -def cosine_cluster -ccselect largest_cluster -defstart 50 -amon

# DnC：dmar=amr 是受控 oracle-budget 消融，不是部署设定
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk minmax -amr 0.20 -adev sign \
  -def dnc -dmar 0.20 -df 1.0 -di 1 -db 10000 -dseed 0 -amon
```

## 结果与审计解释

当启用 `-amon`（任何防御也会自动开启审计）时，输出目录含：

- `round_audit.csv`：每轮选中、上传、接受和屏蔽的客户端 ID，以及恶意/良性计数。
- `client_metrics.csv`：每份防御前上传的 L0、L2、L∞、相对当前样本加权均值更新的余弦相似度，及最终防御决定。
- `group_summary.csv`：按 benign/malicious（仅离线审计真值）分组的均值与标准差。
- `global_metrics.csv` 与 `monitor_dashboard.png`：OA、SA、ASR（仅 label-flip 目标攻击可解释）和检测统计的时间序列。

对 Min-Max、Min-Sum 等 untargeted 更新攻击，不应将零 ASR 解读为防御成功；应报告 clean/attack accuracy、恶意接受率、恶意屏蔽率、良性误屏蔽率以及每轮实际上报人数。`-amr`、`-rmar`、`-dmar` 是不同角色的参数；将它们设成相同数值时，应在论文或报告中标明这是 oracle-budget ablation。

## 致谢与许可证

DFLLib 建立在 [PFLlib](https://github.com/TsingZ0/PFLlib) 的训练框架与部分数据/算法基础设施之上；请在使用或发布基于本项目的结果时同时遵守上游项目的 Apache-2.0 许可证并适当致谢。DFLLib 的安全模块实现与实验审计扩展以本仓库为准。
