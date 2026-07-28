# PFLlib_poison_try：投毒、超参数与 SVD 防御说明

> 本说明以当前工作区代码和两篇随附论文为准：Shejwalkar & Houmansadr（NDSS 2021，以下简称 **SH21**）与 Yang 等（TIFS 2025，以下简称 **Y25**）。它刻意区分「论文提出的原始算法」「本项目已有的实现」和「本次新增的 DnC 复现」。文中给出的命令只是复现实验的配置说明，本文档编写时**没有在 X: 盘运行训练或实验**。

## 1. 这个项目的训练数据流

项目以 PFLlib 为底座；入口是 `system/main.py`。每一次通信轮次的常规 FedAvg 路径是：

```text
全局模型 θ_t
  → select_clients：按 join_ratio 抽样
  → send_models：向所有客户端同步 θ_t
  → 客户端在本地数据上训练 local_epochs
  → receive_models：从被抽中者中模拟掉线/超时，收集上传模型
  → （可选）模型投毒
  → （可选）DnC 检测、屏蔽
  → 按最终留下客户端的样本数加权 FedAvg
  → θ_(t+1)
```

为对齐 SH21 的“单个随机 minibatch 梯度”通信语义，项目另外提供
`-algo FedSGD`：每个选中的客户端只取一个随机 batch，计算一个
stochastic gradient 后上传；服务器加权平均这些梯度，再使用指定的
服务器优化器做一步更新。它不是 `FedAvg -ls 1`：后者仍会遍历本地
DataLoader 的所有 batch。

关键实现位置：

- `flcore/clients/clientbase.py`：数据载入、恶意客户端的标签翻转、SGD 与本地训练共用状态。
- `flcore/servers/serverbase.py`：客户端抽样、掉线、接收、投毒、DnC、防御审计及默认 FedAvg 聚合。
- `flcore/servers/serveravg.py`：FedAvg 的轮次循环；它从 `0` 跑到 `global_rounds`，所以 `-gr 99` 才是 100 轮。
- `flcore/security/attack/minmax_minsum.py`：Min-Max / Min-Sum。
- `flcore/security/monitor/update_monitor.py`：投毒和防御审计 CSV。
- `dataset/<数据集>/config.json`：已生成数据切分的真实 non-IID/IID、类别数、客户端数、样本分布与 `alpha`，优先于命令行上的主观假设。

注意：许多个性化算法的 server 会覆写 `receive_models()` 或 `aggregate_parameters()`。本次 DnC 接入的是 `Server` 的标准接收—聚合路径，FedAvg 可直接使用；对覆写这两个方法的算法，应先检查其是否调用基类逻辑，再谈“该算法已受 DnC 保护”。

## 2. 当前内置攻击到底做了什么

### 2.1 恶意客户端的确定

`-amr/--attack_malicious_ratio` 决定全局被攻陷客户端比例，`-aseed` 用固定随机种子从 `0..num_clients-1` 抽取客户端 ID。这个集合在整个 run 中不变，并保存在 `items/run*/server_malicious_ids.pt`。某一轮是否真的有恶意上传者，还取决于抽样、掉线和超时；它不等于全局恶意集合。

### 2.2 `label_flip`

这是**数据投毒**而非更新投毒。仅当数据集为 MNIST 且客户端已被标记恶意时，`Client.load_train_data()` 用 `LabelFlipWrapper` 改训练标签；当前测试集不改，因而 OA/SA/ASR 仍在干净标签上度量。其有效性取决于标签映射 `utils/label_flip.py` 和恶意客户端本轮是否被选中。

### 2.3 `free_rider`

这是上传阶段的模型替换：恶意上传客户端的参数被独立高斯噪声（当前固定标准差 0.5）覆盖。它并不伪装成正常更新，通常更像“显式异常值”基线；与精心构造的 Byzantine 更新不能直接等同。

### 2.4 `minmax` 与 `minsum`

两者都是**无目标（untargeted）模型更新投毒**，对应 SH21 的 AGR-agnostic attack。客户端先按正常方式本地训练；在 `Server.receive_models()` 内、FedAvg 前，所有本轮活跃恶意客户端的模型更新被替换成同一个优化更新。更新向量定义为：

```text
u_i = flatten(local_model_i - previous_global_model)
u_m = mean(proxy_updates) - λ · deviation
```

这里的 `proxy_updates` 是“本轮真正上传的恶意客户端在替换前的干净本地更新”。这是项目的**攻击者本地代理**，不是 SH21 最强威胁模型中的完整良性更新集合；若恶意客户端的本地分布偏、或本轮仅有一个恶意上传者，代理质量会差很多。`-adev` 决定 `deviation`：

- `unit_vec`：代理均值的单位方向；
- `sign`：代理均值逐坐标符号；
- `std`：代理更新逐坐标标准差（单个代理时为零，攻击将退化为代理均值）。

随后二分/扩张搜索最大的可行 `λ`：

- **Min-Max**：让候选恶意更新到任一代理的平方距离不超过代理两两距离的最大值。它规避“最远距离异常”。
- **Min-Sum**：让候选恶意更新到所有代理的平方距离之和不超过代理样本的参考总距离。它规避“总距离异常”。

SH21 的 AGR-agnostic C 节对两者均明确采用“所有恶意梯度相同”的设定：Min-Max 的式 (6) 写道为最大攻击影响而令所有恶意梯度相同；Min-Sum 的式 (7) 重复同一设定。因此这不是 Min-Sum 特有的实现选择。本项目 `minmax_minsum.py` 也对两种攻击都将同一个搜索得到的 `malicious_update` 写回每一个已上传的恶意客户端；监控中恶意组各指标标准差为零正是这一设定的直接结果。

SH21 的式 (7) 排版为 `max` 参考项；其公开攻击代码与本项目均使用 `min`，因此当前 `minsum` 是“相对最中心代理也不更离群”的较严格/可复现版本。每轮输出的 `lambda/score/threshold` 是搜索过程记录，不是防御检测分数。

## 3. 超参数的设计逻辑：不要把它们混成一个旋钮

| 层次 | 参数 | 控制对象与设计含义 |
|---|---|---|
| 数据分布 | 数据生成脚本、`config.json` 的 `non_iid`、`partition`、`alpha`、`balance` | 先决定每个客户端看见什么数据；它决定“正常更新本应有多分散”，因而直接影响攻击伪装和 SVD 误报。Dirichlet `alpha` 越小通常越偏斜，但应以生成后的 `config.json` 为准。 |
| FL 时间尺度 | `-gr`、`-ls`、`-lr`、`-lbs`、`-ld/-ldg` | 分别是通信轮、每轮本地 epoch、局部 SGD 步长、batch、学习率衰减。`-ls` 或 `-lr` 增大既会增强本地进展，也会拉大客户端漂移；这会同时放大 non-IID 和防御难度。 |
| FedSGD 协议 | `-algo FedSGD`、`-lbs`、`-fopt`、`-flr` | 每客户端每轮仅处理一个 batch；`-fopt {sgd,adam}` 是服务器端对平均梯度做的一步优化，`-flr` 是该步学习率。它用于近似 SH21 的梯度上传接口。 |
| 参与性 | `-nc`、`-jr`、`-rjr`、`-cdr`、`-tsr/-ssr/-tth` | `-jr` 是每轮候选占比，`-cdr` 再从候选中制造掉线，慢客户端/阈值还会使其不上传。防御的实际样本数是“最终上传数”，不是 `nc`，小样本下 SVD 方向更不稳定。 |
| 攻击规模 | `-atk`、`-amr`、`-aseed` | 选攻击、全局恶意比例和固定恶意 ID。报告时要同时写全局比例、每轮上传恶意数和最终被接受恶意数。 |
| Min-Max/Min-Sum 搜索 | `-adev`、`-alinit`、`-atol`、`-amaxsteps` | 选择偏移方向、λ 初始上界、搜索精度、最大迭代数。它们只控制攻击构造，不能当作防御参数。 |
| 审计 | `-amon`、`-amondir`、`-amonplot`、`-amoneps` | 是否记录上传、输出目录、绘图频率、L0 计数阈值。启用 DnC 会自动开启审计，防止“屏蔽了谁”无法追溯。 |
| DnC | `-def dnc`、`-dmar`、`-df`、`-di`、`-db`、`-dseed` | 见第 6 节。最敏感的是未知攻击比例假设 `-dmar`、随机子空间次数/维度，以及 non-IID 程度。 |

个人化算法专用参数（如 `mu`、`lamda`、`K`、`tau`、`beta`）只应在其对应论文/服务器实现中调参；不能拿它们解释 FedAvg+投毒实验。一个可比较的安全实验至少固定：数据切分、模型、`nc/jr/cdr`、`lr/ls/lbs/gr`、攻击比例及随机种子，然后只改变攻击或防御。

## 4. 两篇论文中的 SVD：名字相近，算法不是一件事

| | SH21 的 DnC | Y25 的 MSGuard |
|---|---|---|
| SVD 的角色 | 主防御：利用第一右奇异向量找异常方向，再直接删更新。 | 一个特征：产生 outlier score，和符号统计、余弦分数一起进入 Mean Shift 聚类。 |
| 降维 | 每次随机选 `b` 个坐标，重复 `niters` 次。 | 不用 DnC 的随机坐标作谱检测；从均值梯度中选较大的正、负坐标各一部分（仍另随机抽坐标统计符号）。 |
| 判分 | 居中子更新在首奇异向量上的平方投影。 | 同样的平方投影，但以中位数阈值缩放到 `[0, 0.2]`，只是聚类输入。 |
| 决策 | 每次删除预设 `c·m` 个最大分；多次保留集合取交；平均完整原始更新。 | 先用 L2 范数上下界筛选；在五维特征 `(正/零/负符号数, cosine score, outlier score)` 中取最大 Mean Shift 簇；再与范数筛选集合取交、做裁剪加权平均。 |
| 需要先验 | 需要恶意数量/上界 `m` 和过滤常数 `c`。 | 论文意图是借最大簇避免直接给出恶意数，但引入了范数界、采样比例、坐标数、聚类等新的选择。 |

所以，用户所说的“第一篇还是第二篇的 SVD 防御”若指**单独的 SVD 防御、按论文算法可直接复现**，答案是第一篇 SH21 的 **DnC**。Y25 没有把 SVD 当成单独判决器；它提出的是包含 SVD 的 MSGuard 组合防御。当前代码实现的是 DnC，不应写成“复现 MSGuard”。

### SH21 的 DnC 本质

所以本质上是：把每个客户端更新看成高维点，在一个随机低维子空间中居中后做 SVD；若投毒确实在良性更新云之外形成强主方向，它在首右奇异向量上的投影会大，于是被删掉。

对第 `k` 个随机坐标子集，令 `X` 是 `n × b` 的居中更新矩阵，`v_1` 是其首右奇异向量，则：

```text
score_i = <X_i, v_1>²
```

删掉分数最大的预设数量，重复后对“每次都未被删”的客户端求交，最后平均其**未降维的完整更新**。随机抽样是为让高维模型可算、减少对某一个坐标集的依赖，不是 SVD 本身的安全保证。

### Y25 的 MSGuard 本质

所以本质上是：作者认为单一信号可被 ScaleSign、ALIE 或 Min-Sum 伪装，便让“范数、符号组成、与其他更新的中位余弦、谱离群分数”共同描述客户端，再以最大密度簇当可信集合。它的谱模块有意选大幅正/负均值坐标，以避免 DnC 随机抽坐标漏掉有攻击影响的位置；但这只是论文的设计主张，不能推出它在任意 non-IID 下必然识别。

## 5. SVD 真的能识别吗？尤其在 non-IID 下

不能把“有 SVD”理解成“能鉴定恶意客户端”。它只在某个统计条件成立时提供**离群筛除信号**，不是身份真相。

SH21 的理论条件是良性分布 `B` 与恶意分布 `M` 在首谱方向可分：若两者均值间距相对良性协方差足够大，删除最大投影能高概率去掉恶意更新。直觉上，攻击要造成显著全局偏移，就得引入一个足够明显的共谋方向；这时谱检测有机会抓到它。

但 label-skew/feature-skew non-IID 恰好破坏这个前提：良性客户端可以按标签、域、样本量形成多个天然簇，首主方向可能首先解释“正常数据异质性”，而不是攻击方向。后果有两个方向：

- **误杀**：少数类、少数域或本地 epoch 较多的诚实客户端，可能有很高投影而被屏蔽；这会让模型进一步偏向主流客户端。
- **漏检**：攻击者把更新藏在良性多簇的高方差方向内，或 Min-Max/Min-Sum 把它限制在代理更新的距离范围内，首谱方向不再给出显著分离。

这不是推测：SH21 在 highly non-IID 的 FEMNIST 上明确报告，若最强攻击者知道良性更新，DnC 不能可靠防御；其解释是良性梯度本就不指向单一方向。该文同时报告，在攻击者不知道良性更新的较实际设定下 DnC 仍能缓解攻击。Y25 也把 non-IID 作为更难情形，并报告其 MSGuard 的结果；这是该论文在特定数据切分、攻击和调参下的实证，不是对 DnC 或 SVD 的普遍证明。

因此本项目里 DnC 的正确用途是**可审计的鲁棒聚合基线**：同时报告 clean accuracy、攻击下 accuracy、每轮恶意上传/接受/屏蔽率，以及诚实客户端的误屏蔽率；不要只报告“检测率”。特别应做 IID、pathological non-IID、Dirichlet non-IID 的并列实验，且固定同一套客户端参与和攻击种子。

## 6. 本次接入：忠实的 DnC 核心与可审计记录

新增文件 `flcore/security/defense/dnc.py` 在 `Server.receive_models()` 的攻击之后、FedAvg 之前运行。它只读取上传模型与上一轮全局模型，使用 `update = local - global` 做随机子空间 SVD；**不会读取 `malicious_ids` 来作检测**。`malicious_ids` 只在事后 CSV 中作为实验 ground truth，用来计算漏检/误杀。

实现对应 SH21 Algorithm 2：

1. 在每次迭代随机抽 `min(-db, 参数维度)` 个坐标，居中更新矩阵并作 `torch.linalg.svd`；
2. 以首右奇异向量的平方投影为分数，删除最高的 `round(-df × round(上传数 × -dmar))` 个，至少删一个、至少保留一个；
3. 多迭代时取幸存集合交集；若交集意外为空，保留平均分数最低者并留下该事实的审计痕迹；
4. 对最终幸存客户端重新按样本数归一化，才进入 FedAvg。

DnC 原论文输入包含已知 `m`。真实服务器通常不知道攻击数，所以项目用显式的 `-dmar`（**估计/防御预算，不是真实 `-amr`**）。实验若故意让两者相等，应在论文或日志里写明这是 oracle-budget ablation，不是部署假设。

### AFA-style 余弦聚类基线

项目原先没有 AFA 聚合器或余弦筛选防御。新增 `-def cosine_cluster` 是一个便于对照的 **AFA-style 基线**，并不宣称逐行复现某个 AFA 版本：先按客户端样本数计算本轮防御前加权平均更新 `u_bar`，对每个客户端计算 `cos(u_i, u_bar)`；再仅在这一维分数上做二均值聚类，保留**簇内余弦均值较高**的一簇并重新归一化样本权重聚合。可用 `-cciter` 与 `-cctol` 控制二均值迭代与收敛/退化判据。

它没有 DnC 的随机子空间 SVD，也没有攻击比例预算；因此其前提更强：诚实更新应当占据“与总平均更新更同向”的簇。攻击者若能让自己的更新与平均方向一致、或占比较高从而拉动平均方向，它可能保留攻击者并误删诚实少数簇；non-IID 下尤其必须同时报告误删率。

### 每轮可追溯产物

启用 `-def dnc` 会自动创建 `attack_monitor/run*/`（`-amon` 仍可单独开启普通监控）：

- `client_metrics.csv`：每一个**实际上传、尚未被防御过滤**的客户端，一行；含真实实验组别、L0/L2/L∞、`cosine_to_weighted_mean_update`、`defense_score`、`defense_decision=accepted/blocked`。余弦采用 AFA-style 口径：客户端更新与本轮所有上传更新按样本数加权平均后的余弦，**不再**是完整客户端模型与全局模型的余弦。
- `round_audit.csv`：每一轮一行，保存候选抽样 ID、实际上传 ID、最终接受/屏蔽 ID、各集合的良性/恶意计数、`score_by_client` 与抽样坐标数。列表均为 JSON，便于 pandas 读取。
- `global_metrics.csv`：按评估轮记录 test accuracy、AUC、训练损失、OA、SA 与 ASR；若当前实验未配置目标攻击的 source/target 标签，`asr_available=0`，图中会明确标为不适用，不能把打印出的零 ASR 当成攻击成功率。
- `group_summary.csv` 与 `monitor_dashboard.png`：一张总图汇总准确率/ASR（若可用）、训练损失、良恶更新的 L0/L2/L∞/更新余弦、当前防御分数，以及良恶客户端的接收/屏蔽计数。统计更新均是防御前上传，真实恶意标签仅用于离线审计与作图，不输入防御。

控制台同时打印 `[DnC] accepted=[...] blocked=[...]`。这正好把“本轮被抽到”“上传到了服务器”“被 DnC 接纳并参与最终聚合”三件容易混淆的事分开。

### 建议的起始配置（不在本文档中执行）

从 `system` 目录运行，先做小规模 smoke 配置确认日志字段，再做正式对照：

```bash
python main.py -algo FedAvg -data MNIST -m CNN -ncl 10 \
  -nc 20 -jr 1 -gr 99 -ls 1 -lr 0.005 \
  -atk minmax -amr 0.20 -adev std \
  -def dnc -dmar 0.20 -df 1 -di 1 -db 10000 -dseed 0
```

若要接近 SH21 的 MNIST 梯度协议，而不是普通本地 epoch 的 FedAvg，使用：

```bash
python main.py -algo FedSGD -data MNIST -m DNN -ncl 10 \
  -nc 100 -jr 1 -gr 499 -lbs 100 \
  -fopt adam -flr 0.001 \
  -atk minsum -amr 0.20 -adev sign -amon
```

这里 `FedSGD` 的 Adam 状态保存在服务器端，作用于每轮加权平均的梯度；
这比让每个客户端各自维护 Adam 状态后再平均模型更接近同步梯度训练。论文的
MNIST FC 为 `{784,512,10}`，而当前 `DNN` 的隐藏维度为 100；完整网络结构、初始化、数据切分和其它 AGR 仍需逐项对齐，不能仅凭该命令声称
已数值复现论文。

至少比较四组：无攻击+FedAvg、攻击+FedAvg、攻击+DnC（`dmar=amr` 的受控上界）、攻击+DnC（错设 `dmar`）。然后只替换数据切分做 IID/Dirichlet/pathological non-IID；每组用多个种子，报告均值和方差。对于很小的每轮上传数，不应设置很大的 `-di`：交集会快速缩小，这不是“防御更强”，而是保留集被过度收缩。

## 7. 当前边界与下一步

1. 目前只复现了 SH21 的 DnC，不是 Y25 的 MSGuard；后者需要额外实现坐标选择、符号统计、余弦缩放、Mean Shift 与裁剪加权，且需要独立复现其超参数表。
2. 当前 Min-Max/Min-Sum 使用恶意客户端自身的干净本地更新做代理，属于明确的较弱/本地攻击知识设置；请勿把结果表述为 SH21 全知攻击者结果。
3. 当前 `label_flip` 在代码中只对 MNIST 生效；在其他数据集上指定该攻击不会产生同等语义的标签映射。
4. 防御分数是“本轮、该随机子空间、该参与集”中的相对异常分数，不能跨轮直接比较，也不能当作恶意概率。

若要回答“非 IID 下能否识别”，建议以 `round_audit.csv` 为主表，对每轮计算：恶意接受率、恶意屏蔽率、诚实误屏蔽率、最终聚合的恶意权重，以及测试精度/各类精度。只有这些指标与数据偏斜度一起看，才能区分“准确率下降是攻击成功”还是“防御误杀了异质但有用的诚实客户端”。
