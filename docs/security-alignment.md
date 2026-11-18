# 两个子库的攻防功能对齐

验证日期为 2026-11-18。

`DFLlib` 用于对外分发，`PFLlib-master` 用于本地实验。两库各自包含可独立运行的攻防源码、训练入口、评估工具和测试。迁移以两边当前工作树的功能并集为基础。

## 功能清单

| 范围 | 两库共有的功能 |
|---|---|
| 更新攻击 | Label Flipping、Gaussian Free-Rider、Min-Max、Min-Sum、Fang-Krum、PoisonedFL |
| Min-Max 变体 | `agnostic`、`cosine`、`cosine_oracle` |
| 图像后门 | BadNets、Blended、Refool 风格、Clean-label、DBA、Neurotoxin、A3FL、F3BA |
| 后门别名 | `reflect` 对应 `refool`，`label_consistent` 对应 `clean_label` |
| 鲁棒聚合 | Krum、Multi-Krum、Bulyan、Trimmed Mean、Coordinate Median、DnC、Cosine Cluster |
| 标量防御 | `scalar_kmeans`、`scalar_dbscan` |
| 原型方法 | `proto-clean`、`proto-probe` |
| 离线修复与评估 | PGBD、参考集制作、模型干净准确率与触发 ASR 评估 |
| 审计 | 上传与接受集合、更新统计、攻击轨迹、标量特征、OA/SA/ASR |

PoisonedFL、标量防御和图像后门的完整训练路径覆盖 FedAvg/FedSGD。FedSGD 支持服务器 SGD 和 Adam。其他个性化算法可覆盖不同的接收与训练方法，使用前需要逐项确认调用路径。

## 保留的实验设定

- Min-Sum 保留两库原有的 `max` 代理距离平方和阈值。测试显式核验此行为。
- PoisonedFL 的默认反馈窗口为 10。使用 `-pfw 50` 可选择论文窗口。FedSGD/Adam 使用显式的 `adam_proxy` 梯度映射。
- Refool 保留镜像混合近似。Clean-label 保留目标类样本加触发器的轻量基线。A3FL、F3BA 保留项目现有的简化预设。
- `proto-probe` 记录诊断分数并保留全部客户端。PGBD 使用带标签的干净参考集。

完整参数说明见 [安全模块说明](../system/flcore/security/README.md)。

## 本次修复

- 两库统一客户端投毒包装、DBA 联盟分片、通信轮次同步、历史全局参数及服务器触发测试入口。
- PoisonedFL 的恶意客户端跳过私有训练，三个上传映射共用同一实现。
- 通信轮次独立于监控开关推进。投毒比例为 0 时保持零投毒样本。
- A3FL 在触发器优化时保留原始源标签，随后对训练批次投毒。影子模型的更新运行在梯度上下文内。
- A3FL 的训练中 ASR 使用实际学到的各客户端图案。保存全局模型时，图案和配置写入配套的 `*_backdoor.pt`。
- 离线模型评估使用批处理输入，并分别计算完整测试集的 `clean_accuracy`、符合源类条件的 `source_accuracy` 及触发器 ASR。
- 旧监控测试已接入当前上传权重和加权平均更新余弦接口。Min-Sum 的旧测试与说明同步到保留的 `max` 阈值行为。

## 验证结果

两个库各执行完整测试集，共 64 项，63 项通过，1 项因当前环境缺少 CUDA 而跳过。原有测试中的两项接口或说明失配已经修正。

从各库的 `system/` 运行以下命令。

```bash
python -m unittest discover -s tests -v
```

新增的后门集成测试覆盖八种预设在 FedAvg/FedSGD 中的实际客户端训练、接收及聚合。还覆盖投毒日程、零投毒比例、保留标签、DBA 分片、梯度屏蔽、权重翻转、A3FL 图案保存及评估。

每库另完成 19 组两轮训练入口冒烟验证，使用随机合成的 MNIST 形状小型数据分区，每组固定八个客户端、两个恶意客户端。覆盖六组旧更新攻击或基线、PoisonedFL 的三个上传映射、两个余弦 Min-Max 变体和八种图像后门。BadNets 与 A3FL 各完成一次保存模型的离线评估。该验证支持迁移与接口正确性的结论。攻击效果和单教师蒸馏迁移率由后续专门实验测量。

`test_security_alignment.py` 比较两库的 25 个安全模块及工具源码、5 个客户端与服务器入口文件，以及安全参数和校验函数。对外分发单个仓库时，该跨库检查自动跳过，仓库自身的功能测试照常执行。

## 保存模型评估

从 `system/` 运行工具，并传入本地模型及对应数据集参数。

```bash
python -m tools.evaluate_backdoor_model --help
python -m tools.run_pgbd_repair --help
```

A3FL 模型评估时通过 `--trigger-state` 指向配套的 `*_backdoor.pt`，工具从中恢复触发器配置。多个图案按图像与图案的组合等权汇总 ASR。方块 BadNets 使用与训练一致的目标类、patch 大小和位置参数。
