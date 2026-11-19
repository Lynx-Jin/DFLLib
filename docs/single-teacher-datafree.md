# 单教师 BadNets 与生成器无数据蒸馏

先监督训练CIFAR-10教师，再从随机权重初始化同架构学生，只用生成器合成图像蒸馏。教师与学生均为框架的`FedAvgCNN(3, 10, 1600)`，不进入联邦聚合。

## BadNets 预设

| 设置 | `reference`（默认） | 单教师脚本的`legacy` |
| --- | --- | --- |
| 投毒数量基准 | 全部训练记录的20% | 非目标训练记录的10% |
| 可选记录 | 非目标类别 | 非目标类别 |
| 触发器 | 右下角4×4白块，边距1 | 左上角5×5白块，边距0 |
| 目标类别 | 0 | 0 |

Reference使用固定的客户端种子和NumPy RandomState索引；正比例、可选记录非空时至少选一条，数量不超过可选记录数。原生联邦CLI的legacy默认比例仍为25%，与本单教师脚本显式保留的10%历史配置区分。

改变预设时重新训练教师，不能用旧触发器训练的教师检查点评估新的触发器。配置和教师元数据会记录预设，恢复时检查一致性。

## 一键运行

先准备官方CIFAR-10原始批次：`DFKD_DATA_ROOT`目录下应包含`cifar-10-batches-py/data_batch_1`至`data_batch_5`及`test_batch`。使用已安装本框架依赖、支持CUDA的Python环境。在仓库根目录执行：

```bash
cd system
export DFKD_PYTHON=python
export DFKD_DATA_ROOT=/path/to/cifar10/rawdata

bash tools/reproduce_single_teacher_datafree.sh badnets
bash tools/reproduce_single_teacher_datafree.sh clean
```

两条命令独立运行。默认均训练80轮CNN教师，之后执行6,000个蒸馏循环，每循环1次生成器、10次学生更新。`badnets`使用reference投毒，`clean`不投毒；两者均用相同触发器定义评估ASR。采用Tanh生成器、学生Adam 0.0002、生成器Adam 0.001、KL加250倍空间注意力损失。`--amp`仅用于教师训练，蒸馏保持float32。

可指定第二个参数作为输出目录，已有实验目录会拒绝覆盖。默认输出位于被Git忽略的`results/reproduce_*`，包含实际命令、配置、日志、逐次指标和权重。只查看命令可用：

```bash
DFKD_DRY_RUN=1 bash tools/reproduce_single_teacher_datafree.sh badnets
DFKD_BACKDOOR_PRESET=legacy bash tools/reproduce_single_teacher_datafree.sh badnets
```

第二条会按legacy预设实际启动训练。更细配置使用：

```bash
python -m tools.run_datafree_badnets --help
```

Python入口支持`--phase teacher`、`--phase distill`和`--phase full`；已有教师时可在原输出目录使用`--phase distill --resume`。恢复必须沿用相同配置和总预算，不通过改变预算隐式延长已有余弦计划。

## ZSKT 适配范围

参考[ABD官方ZSKT实现](https://github.com/illidanlab/ABD/tree/19c89bf5413ab092aef214ae8a6ef69aae08be71/zskt)。Python入口的`--distillation-preset abd-zskt`使用80,000循环、每循环1次G与10次S更新、128个100维高斯噪声、G Adam 0.001、S Adam 0.002和余弦调度；显式参数可覆盖默认值。其生成器末端为affine BatchNorm，不加Tanh，每次学生更新重新前向生成器和教师，保留基线的BN更新次数。

一键脚本使用`custom`预设的CNN适配：Tanh限制合成输入范围，降低学生学习率，保留KL和空间注意力目标。它不是官方WRN架构的逐项复现，也没有启用ABD的Shuffling Vaccine或Self-Retrospection防御。KL以数学等价的log-target形式计算，避免极端logits下的概率下溢。

## 数据边界、指标与健康检查

教师训练使用官方50,000张训练图像。蒸馏更新不接收真实训练图像或原始标签，不注入触发器；官方10,000张测试图像仅用于无梯度评估。教师冻结，结束时验证权重哈希。

同时查看clean accuracy、非目标样本ASR、clean target rate、trigger lift和原本分类正确样本的条件ASR。学生任务准确率很差或偏向目标类时，单看ASR不能证明后门传递或去除。

每次评估记录`synthetic_health_*.json`：合成像素范围、师生预测分布、一致率、隐藏层零激活比例及logits变化。若连续三次评估发现至少99%合成样本隐藏层全零且输出几乎恒定，默认保存检查点、写入`training_failure.json`并停止。可显式设置`--collapse-patience 0`关闭此检查。

仓库发布源码、测试和协议说明，不包含实验日志、数据集或模型权重。
