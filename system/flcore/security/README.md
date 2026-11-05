# `flcore/security` 攻防实现说明

## DFLlib 新增攻防入口

完整机制、默认值、信息边界及可执行对照命令见
[`scalar-security-port.md`](../../../docs/scalar-security-port.md)。

- `-atk poisonedfl -pfs 8 -pfw 10`：只使用全局模型历史的基础共享载荷攻击。
  FedAvg提交模型差分；FedSGD/SGD使用`-pfgm sgd_delta`；
  FedSGD/Adam显式使用`-pfgm adam_proxy`，这是梯度代理，不是Adam精确逆换算。
- `-atk minmax -amv cosine_oracle -acq .5`：使用恶意干净代理的模型/全局余弦
  分位数约束，替换原L2可行性约束；不读良性更新或聚类决策。
  `-amv cosine`为保留L2上界的版本，默认`agnostic`仍走既有实现。
- `-def scalar_kmeans`：对`cos(P_i,W)`二聚类，保留人数多数簇，平局取高余弦簇。
- `-def scalar_dbscan`：相同阶段一后，仅在主簇内用`||P_i||`和`cos(P_i,W)`
  检查紧密DBSCAN子簇及客户端/伙伴历史。后端不接收向量、数据、真恶意标签或比例。
- `-nc 20 -amr .2 -aseed 0`：总20人中4恶意16良性，不追加客户端。
- `-defstart 5`前不实际剔除，双标量方法仍收集主簇合格历史；全拒绝时保持全局
  模型，且不推进Adam状态。
- `-sclog`打印排序的每客户端模型L2/余弦；聚类和两阶段剔除写入实验目录
  `security/runN/submitted_scalars.jsonl`；攻击参数写入`attack_trace.jsonl`。

现有`cosine_cluster`及监控余弦参照加权平均**更新**；新方法参照轮次开始的
**全局模型**。FedSGD的提交P为原有`W+g`载体。新旧指标不混用，亦不将新组合
称为FLAME原样复现。原有方法说明保留如下。

本目录是项目所有投毒、鲁棒聚合与审计逻辑的唯一入口说明。每次新增或改变 `attack/`、`defense/`、`monitor/` 的行为时，应同步更新本文件；日志中的真恶意标签只用于离线审计，任何防御实现均不得读取它。

## 统一数据流

每轮客户端上传模型后，服务器以 `u_i = local_model_i - global_model` 作为更新。攻击先改写恶意上传，防御随后读取防御前上传并做决策，最后才聚合。`round_audit.csv` 保存上传、接受和屏蔽集合；`client_metrics.csv` 保存防御前更新统计；逐坐标防御的客户端状态会标为 `coordinate_aggregated`，因为它们不以“删整客户端”的方式工作。

## 攻击

| `-atk` | 逻辑 | 重要前提 |
|---|---|---|
| `none` | 不修改上传。 | — |
| `label_flip` | 恶意 MNIST 客户端训练时翻转标签。 | 数据投毒；不是任意更新攻击。 |
| `free_rider` | 将恶意上传参数替换为高斯噪声。 | 显式异常基线，通常不隐蔽。 |
| `minmax` | 用恶意客户端本地干净更新做代理，搜索共享更新，使其到任一代理的最大平方距离受代理两两最大距离约束。 | SH21 AGR-agnostic 风格；不知服务器 AGR，也不直接约束余弦/L∞。 |
| `minsum` | 同上，但约束到所有代理的平方距离和。 | 本实现遵循公开代码使用最中心代理的 `min` 参考项。 |
| `fang` | 以全部本轮良性上传为参考，用反符号方向搜索仍会被 Krum 选中的最大共享更新。 | **oracle/AGR-tailored Fang-Krum**：攻击者可见良性更新且知道 Krum 参数；不能称为 agnostic。 |

`minmax` 与 `minsum` 都令当轮全部恶意上传为同一个优化更新；这是 SH21 C 节式 (6)、式 (7) 的共同设定。`fang` 当前只实现 Krum 目标，使用 `-fmar` 指定攻击者已知的 Krum Byzantine 预算。

常用攻击参数：`-amr` 恶意客户端比例，`-aseed` 恶意 ID 种子，`-adev {unit_vec,sign,std}`，`-alinit`、`-atol`、`-amaxsteps` 搜索参数，`-fmar` Fang-Krum 的已知预算。

## 攻击参数手册（默认值与命令）

### 所有攻击共用的参数

| 指令 | 默认值 | 作用 | 适用攻击 |
|---|---:|---|---|
| `-atk` / `--attack_type` | `none` | 选择攻击类型。 | 全部 |
| `-amr` / `--attack_malicious_ratio` | `0.0` | 从全局 `nc` 个客户端中固定抽取的恶意客户端比例；每轮实际恶意上传数仍受 `jr`、掉线和超时影响。 | 除 `none` 外全部 |
| `-aseed` / `--attack_seed` | `0` | 恶意客户端 ID 集合的 NumPy 种子；同一 run 内 ID 集合不变。 | 除 `none` 外全部 |
| `-atk_ex_eval` | `False` | 是否在全局指标评估时排除恶意客户端的测试数据。当前参数使用 `type=bool`，命令行显式传值容易产生歧义；正式比较建议保持默认并在报告中说明。 | 全部 |
| `-amon` | 关闭 | 启用 `attack_monitor/run*/` CSV 与总图；任何防御也会自动开启审计。 | 全部 |
| `-amondir` | `attack_monitor` | 监控输出根目录。 | 全部 |
| `-amonplot` | `10` | 每 N 轮刷新一次 dashboard。 | 全部 |
| `-amoneps` | `0.0` | L0 统计中认为坐标非零的绝对值阈值；不改变攻击或聚合。 | 全部 |

### `none`

没有投毒。推荐仍显式写 `-atk none -amr 0`，作为与攻击组完全对齐的 clean 对照。

```bash
python main.py -algo FedSGD -data MNIST_100 -m FedAvgMLP -nc 100 -jr 1 -gr 500 -lbs 100 \
  -atk none -amr 0 -amon
```

### `label_flip`

当前代码对名称以 `MNIST` 开头的所有数据集版本有效，例如 `MNIST`、`MNIST_100`、`MNIST_100_unbalanced`；把恶意客户端训练标签 `7 -> 1`，测试集不翻转。没有 source/target 或映射的命令行参数；映射固定在 `utils/label_flip.py`，且 OA/SA/ASR 的 source/target 统计使用同一前缀判断。

```bash
python main.py -algo FedAvg -data MNIST -m CNN -nc 100 -jr 1 -gr 500 -ls 1 \
  -atk label_flip -amr 0.20 -aseed 0 -amon
```

### `free_rider`

恶意客户端完成本地训练后，其**整份上传模型参数**被独立高斯噪声覆盖。噪声标准差目前硬编码为 `0.5`，没有 `-free_rider_std` 参数；若研究需要扫强度，应先把该常数显式参数化，再记录配置。

```bash
python main.py -algo FedAvg -data MNIST_100 -m FedAvgMLP -nc 100 -jr 1 -gr 500 -ls 1 \
  -atk free_rider -amr 0.20 -aseed 0 -amon
```

### `minmax` 与 `minsum`

| 指令 | 默认值 | 含义 |
|---|---:|---|
| `-adev` | `std` | `unit_vec` 为代理均值单位方向，`sign` 为代理均值逐坐标符号，`std` 为代理更新逐坐标标准差。论文 MNIST 常应至少对照 `sign`；不要把 `std` 当作必然最强。 |
| `-alinit` | `10.0` | λ 搜索的初始上界；若仍可行，代码会自动扩张。 |
| `-atol` | `1e-5` | 二分搜索停止精度。 |
| `-amaxsteps` | `80` | 扩张与二分使用的最大迭代预算。 |

这些参数只控制“共享恶意更新”的构造，不是防御阈值。`minmax` 的可行性比较最大平方距离，`minsum` 比较距离平方和；它们均只使用当轮恶意客户端替换前的本地干净更新做代理。

```bash
# 论文 MNIST 协议方向的 Min-Max / sign 示例
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk minmax -amr 0.20 -aseed 0 \
  -adev sign -alinit 10 -atol 1e-5 -amaxsteps 80 -amon

# Min-Sum / std 示例（std 是默认值，仍建议显式写出）
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk minsum -amr 0.20 -aseed 0 \
  -adev std -alinit 10 -atol 1e-5 -amaxsteps 80 -amon
```

### `fang`

| 指令 | 默认值 | 含义 |
|---|---:|---|
| `-fmar` | `0.20` | Fang 攻击者已知的 Krum Byzantine 预算 `f/n`。应与目标 Krum 的 `-rmar` 对齐，才能构成已知防御的受控评估。 |
| `-alinit` | `10.0` | 反符号偏移 λ 的初始搜索上界。 |
| `-atol` | `1e-5` | λ 搜索精度。 |
| `-amaxsteps` | `80` | 最大搜索迭代预算。 |

当前 Fang 使用良性上传的真实均值和反符号方向，搜索“仍被 Krum 选中”的最大 λ；若连 λ=0 的参考点也无法被 Krum 选中，则该轮不覆盖恶意上传。因此它只应与 `-def krum` 配对解释。

```bash
python main.py -algo FedSGD -fopt adam -flr 0.001 -data MNIST_100 -m FedAvgMLP \
  -nc 100 -jr 1 -gr 500 -lbs 100 -atk fang -amr 0.20 -aseed 0 \
  -fmar 0.20 -alinit 10 -atol 1e-5 -amaxsteps 80 -def krum -rmar 0.20 -amon
```

## 防御

| `-def` | 处理方式 | 关键参数与约束 |
|---|---|---|
| `none` | 普通样本数加权 FedAvg；FedSGD 则对平均梯度做服务器优化器更新。 | — |
| `krum` | 计算每个更新到最近 `n-f-2` 个邻居的平方距离和，仅聚合最小者。 | `-rmar` 给出 `f/n`；要求 `n >= 2f+3`。 |
| `multikrum` | 迭代选 Krum 胜者，再平均入选更新。 | `-mks 0` 自动取最大合法选择数 `n-2f-3`；同样要求 `n >= 2f+3`。 |
| `bulyan` | 先用 Multi-Krum 选 `n-2f` 个候选；每一坐标保留最接近候选中位数的 `n-4f` 个值并平均。 | 要求 `n >= 4f+3`；例如 `n=100,f=25` **不合法**。 |
| `trimmedmean` | 每一坐标排序，删去两端各 `f` 个值，平均中间值。 | 要求 `n > 2f`。 |
| `median` | 每一坐标取中位数。 | 使用 `-rmar` 统一记录预算；不依赖客户端整体筛除。 |
| `cosine_cluster` | 对 `cos(u_i, sum_j w_j u_j)` 做一维二均值聚类；可按余弦均值或簇大小选择保留簇。 | AFA-style 基线，不是 AFA 的精确复现；`-ccselect {high_mean,largest_cluster}`、`-cciter`、`-cctol`。 |
| `dnc` | 随机抽坐标、中心化、SVD；删除首右奇异向量投影平方最大的更新。实现直接从参数张量 gather 抽样坐标，不先构造完整客户端更新矩阵。 | `-dmar`、`-df`、`-di`、`-db`、`-dseed`。 |

`-rmar` 是 Krum/Multi-Krum/Bulyan/Trimmed Mean/Median 的公开防御预算；`-dmar` 是 DnC 的独立预算。两者均不是攻击真值 `-amr`。实验若使其相等，需标注为 oracle-budget ablation。

所有 `-def` 共用 `-defstart` / `--defense_start_round`（默认 `0`）：仅当通信轮编号 `round >= defstart` 时，服务器才调用所选防御。设为 `50` 表示第 0--49 轮按普通聚合运行、第 50 轮起开始防御；warm-up 期间监控仍记录上传与 audit，但 defense 字段为 `none`。该参数不改变攻击开始时刻，攻击仍从第 0 轮起按配置执行。

DnC 的 `-db` 是每次 SVD 的随机坐标数，不是完整模型维度。对 100 客户端 ResNet18/VGG11，建议先保持 `-db 10000 -di 1`；增大 `-db` 会增加 SVD 时间与 CPU 内存。当前实现不会再先展平全部客户端模型更新，但 `-db` 过大仍会使 SVD 本身变慢。

## 命令示例

以下从 `system/` 目录运行；示例只说明组合，不代表已运行或推荐的正式实验设置。

```bash
# 无 AGR 知识的 Min-Max 对 Krum
python main.py -algo FedSGD -data MNIST_100 -m FedAvgMLP -nc 100 -jr 1 -gr 500 -lbs 100 \
  -atk minmax -amr 0.20 -adev sign -def krum -rmar 0.20 -amon

# Oracle Fang-Krum：攻击方与防御方采用同一已知 Krum 预算
python main.py -algo FedSGD -data MNIST_100 -m FedAvgMLP -nc 100 -jr 1 -gr 500 -lbs 100 \
  -atk fang -amr 0.20 -fmar 0.20 -def krum -rmar 0.20 -amon

# AFA-style 余弦聚类基线
python main.py -algo FedSGD -data MNIST_100 -m FedAvgMLP -nc 100 -jr 1 -gr 500 -lbs 100 \
  -atk minsum -amr 0.20 -adev sign -def cosine_cluster -ccselect high_mean -cciter 50 -cctol 1e-6 -amon

# DnC；这里 dmar=amr 仅是受控 oracle-budget 消融
python main.py -algo FedSGD -data MNIST_100 -m FedAvgMLP -nc 100 -jr 1 -gr 500 -lbs 100 \
  -atk minmax -amr 0.20 -adev sign -def dnc -dmar 0.20 -db 10000 -amon
```

## 报告要求

至少同时报告 clean accuracy、攻击后 accuracy、恶意接受率、恶意屏蔽率、良性误屏蔽率及每轮实际上传人数。对于 `trimmedmean`、`median`、`bulyan`，还应标注其为坐标级处理，不能将 audit 中的“accepted 客户端数”误解释为所有参数均完整参与聚合。对 untargeted 攻击，ASR 没有语义；应报告 accuracy drop，而不是把零 ASR 当作安全结论。
