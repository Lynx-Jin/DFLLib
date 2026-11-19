# BadNets 与联邦多教师 ZSKT

原生`FedAvgZSKT`流程为：下发全局模型 → 客户端本地训练/BadNets → 按样本量FedAvg得到student → 记录KD前指标 → 冻结上传模型作为teacher pool → 生成器辅助ZSKT → 记录KD后指标 → 下轮广播。

## 默认协议

| 设置 | 默认值 |
| --- | --- |
| 客户端 | 20个，全参与；20%恶意客户端，attack seed 0 |
| 模型 | 所有教师和学生使用同一FedAvgCNN，878,538参数 |
| 本地训练 | 每轮2 epochs，batch 10，SGD 0.005恒定，无动量、权重衰减和在线增强；drop_last=True |
| 模型/训练种子 | 1000；攻击种子独立设为0 |
| BadNets | `reference`：全部本地记录20%计数，从非目标记录中选择，跨轮固定索引 |
| 触发器 | 右下角4×4白块，边距1，目标0；ASR排除目标类别 |
| 联邦轮次 | 200次本地训练与聚合更新 |
| 每轮服务器KD | 200循环，每循环1次G、10次student更新；合成batch 128 |
| KD目标 | 样本量加权的教师类别概率混合；KL + 250倍空间注意力 |
| KD优化 | student Adam 0.0002、G Adam 0.001，温度1 |

原生浮点FedAvg没有包含其他工程的定点编码、截断或解码聚合链路。参数协议相同不意味着不同数值后端的实验指标逐位相同。

## 数据准备

启动器默认`FL_ZSKT_DATA_PROTOCOL=legacy`，复用已有CIFAR-10客户端训练/留出记录，不重新抽样。数据目录须包含：

- `config.json`：`num_clients=20`、`num_classes=10`及划分元数据。
- `train/0.npz`至`train/19.npz`、`test/0.npz`至`test/19.npz`。
- 每个npz的`data`为含`x`和`y`的字典；`x`为归一化到`[-1,1]`的NCHW图像，`y`为类别0至9。

`FL_ZSKT_LEGACY_DATA_ROOT`可明确指定来源；默认会优先复用相邻PFLlib-master的相应数据，再查找本仓库缓存。准备工具逐文件校验源记录和缓存的SHA256，不覆盖不匹配的缓存。使用同一数据源运行两个方法，确保划分与评估集合一致。原始数据、客户端分片和生成缓存均不随仓库发布。

也可选择`FL_ZSKT_DATA_PROTOCOL=official`：从`DFKD_DATA_ROOT/cifar-10-batches-py`读取官方50,000训练/10,000测试记录，生成类别均衡IID分片。此协议与历史客户端留出协议不同，目前不支持Dirichlet模式。两种协议的真实记录都只用于客户端训练和服务器评估，服务器KD更新仅使用生成器输出。

## IID 对照

从仓库根目录执行，使用已安装依赖的CUDA环境，并将数据路径替换为实际位置：

```bash
cd system
export DFKD_PYTHON=python
export FL_ZSKT_DATA_PROTOCOL=legacy
export FL_ZSKT_PARTITION=iid
export FL_ZSKT_LEGACY_DATA_ROOT=/path/to/Cifar10_iid_clients
unset FL_ZSKT_RESUME FL_ZSKT_LOGGER_ROOT

# 相同BadNets和本地训练，关闭服务器KD
bash tools/reproduce_federated_zskt.sh fedavg

# 相同设置，FedAvg后增加多教师ZSKT
bash tools/reproduce_federated_zskt.sh badnets
```

默认日志根目录为`results/federated_zskt_badnets_reference/`，两组各自创建独立实验目录。可选`clean`模式将恶意客户端比例置为0并保留KD。不能用不同投毒定义或不同划分的检查点继续实验。

## Dirichlet α=0.1 对照

复用已有α=0.1分片；源`config.json`必须包含`non_iid=true`、`partition="dir"`、`alpha=0.1`，否则在训练前报错。从`system/`执行：

```bash
export DFKD_PYTHON=python
export FL_ZSKT_DATA_PROTOCOL=legacy
export FL_ZSKT_PARTITION=dirichlet
export FL_ZSKT_ALPHA=0.1
export FL_ZSKT_LEGACY_DATA_ROOT=/path/to/Cifar10_dir0.1_clients
unset FL_ZSKT_RESUME FL_ZSKT_LOGGER_ROOT

bash tools/reproduce_federated_zskt.sh fedavg
bash tools/reproduce_federated_zskt.sh badnets
```

默认缓存名为`dataset/Cifar10_ZSKT_dir0.1_legacy20/`，日志根目录为`results/federated_zskt_badnets_reference_dir0.1/`；实验goal包含`dir0.1`，协议保存`data_partition`和划分指纹。默认仍为IID，在同一终端切回时需设置`FL_ZSKT_PARTITION=iid`并切换数据源。

20%指恶意客户端数量比例。非IID样本量不均衡时，其样本量加权聚合/教师权重不一定等于20%；每轮记录实际`malicious_teacher_weight`。投毒计数仍基于各恶意客户端全部本地记录，并由非目标可选记录数量封顶。跨划分比较时同时报告该权重、实际投毒数和非目标测试分母。不能预设KD在异质性下提高精度或ASR，应通过对照实验检验。

## 记录与归因

教师目标为按样本量平均的类别概率，注意力为相同权重下的归一化空间响应均值；不用恶意身份过滤教师。G跨轮保留，student每轮重建Adam。KD随机状态与客户端训练隔离。服务器蒸馏不使用真实训练图像、原始标签或人工注入触发器；留出记录只做no-grad评估，已归一化输入不会重复归一化。

每个实验的`zskt/run0/`包含：

| 文件 | 含义 |
| --- | --- |
| `protocol.json`、`command.json` | 实际协议、数据指纹、执行命令和数值后端 |
| `metrics.csv` | `fedavg_before_kd`、`teacher_ensemble`、`after_kd`；无KD时为`no_kd_control` |
| `rounds.jsonl` | 教师表现、实际权重、客户端投毒数量/索引哈希、教师冻结校验 |
| `status.json`、`kd_progress.json` | 当前轮次和KD进度 |
| `global_model.pt` | 最近完成轮的全局模型 |
| `fedavg_student_latest.pt`、`uploaded_teachers_latest.pt` | KD前student和上传教师 |
| `checkpoint.pt` | 模型、G、优化器、调度器和随机状态，供同协议恢复 |
| `source/`、`source_sha256.json` | 实际执行源码及哈希，包括数据准备和启动脚本 |

FedAvg本身可能传入后门。比较完整FedAvg/KD实验的最终clean ACC和ASR，并查看同轮KD前后变化，以区分整体训练轨迹和当轮蒸馏效果。同时报告clean target rate、trigger lift及条件ASR，避免将随机初始化的目标类偏置解释为后门传递。

## 覆盖、恢复与历史预设

```bash
# 只查看准备和训练命令
DFKD_DRY_RUN=1 bash tools/reproduce_federated_zskt.sh badnets

# 同配置、同总预算、同数据指纹恢复
FL_ZSKT_RESUME=/path/to/zskt/run0/checkpoint.pt bash tools/reproduce_federated_zskt.sh badnets
```

可通过`FL_ZSKT_ROUNDS`、`FL_ZSKT_LOCAL_EPOCHS`、`FL_ZSKT_KD_CYCLES`、`FL_ZSKT_MALICIOUS_RATIO`等变量显式覆盖默认值；实际值写入协议。改变总预算、划分或攻击配置时重新运行，不沿用不匹配的检查点。

原生`main.py --backdoor badnets`默认reference。显式`--backdoor_preset legacy`保留按非目标记录比例计数、跨轮变化的索引、左上角5×5且无边距的旧行为；其他攻击方法保留原语义。单教师入口说明见[单教师指南](single-teacher-datafree.md)。
