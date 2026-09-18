# 第二轮实验：改善近端预测，验证有效表征

针对上一轮 NREL 结果，本轮优先检查近端预测误差、输出越界和探针代表性，再验证较小编码宽度是否足够。默认仍关闭重构、方差和协方差正则。代码仅完成静态检查，**没有在本机执行 Python、测试、训练或图表渲染**；下列命令均交给目标服务器执行。

## 1. 更新和服务器最小检查

在已有仓库中更新并激活原来的 Python 环境：

```bash
git pull --ff-only
cd forecast_lab
python -m pip install -r requirements.txt
python server_checks.py
```

新增检查涵盖残差目标列顺序、编码/最后值两条路径的干预、近端加权与缺失掩码、原始单位的范围统计、数据文件变更检测。检查命令已提供，但尚未执行；不能把它当成测试已通过的声明。

先用服务器上的一个实际 CSV 检查完整流程。将路径改成真实文件：

```bash
python train.py --config configs/nrel_v2.json \
  --data /srv/data/NREL_WTK_site101275_your_file.csv \
  --output runs/v2_server_smoke --device cuda:0 \
  --lookback 32 --horizon 24 --d-model 8 --layers 2 \
  --epochs 2 --probe-windows 16 \
  --prediction-mode residual --near-weight 4
```

确认生成 `report.html`、`figures/*.png` 和 `summary.json` 后，再开始正式实验。最小运行的分数不用于方法比较。

## 2. 四组受控对照

| 名称 | 预测方式 | 前 12 步损失权重 | 其他步权重 |
|---|---|---:|---:|
| baseline | 直接预测未来 | 1 | 1 |
| residual | 最后观测值 + 网络预测的修正量 | 1 | 1 |
| near | 直接预测未来 | 4 | 1 |
| residual_near | 最后观测值 + 网络预测的修正量 | 4 | 1 |

各组使用相同数据划分、编码宽度、训练预算与种子集合。残差模式不增加可训练参数，也不另行改变预测头初始化。`near_weight=4` 只是预先指定的实验值，不代表已调优。

损失为 `sum(mask * lead_weight * squared_error) / sum(mask * lead_weight)`。近端长度是 `min(near_steps, horizon)`；如果所有预测步都属于近端，该权重归一化后与普通 MSE 等价。以 5 分钟数据为例，12 步代表 1 小时，H=24/96/244 分别代表 2/8/20 小时 20 分钟。

所有组的早停、学习率调整、最佳检查点选择均使用**未裁剪、未加权的完整验证 MSE**。同时记录第一步、前 12 步、其余步的指标，观察近端改善是否牺牲远端。`train_prediction` 是未加权 MSE，`train_weighted_prediction` 是实际优化的预测损失。

四站点入口沿用上一轮服务器数据目录，可通过环境变量覆盖：

```bash
# 先打印计划；不训练，不创建运行目录（仍需文件存在）
DATA_DIR=/srv/data/nrel DRY_RUN=1 bash run_nrel_v2.sh

# 第一批：4 站点 × 1 个预测长度 × 1 个种子 × 4 组 = 16 次训练
DATA_DIR=/srv/data/nrel SEEDS="42" HORIZONS="24" bash run_nrel_v2.sh

# 正式：4 站点 × 3 个预测长度 × 3 个种子 × 4 组 = 144 次训练
# 已完整完成且配置、数据指纹一致的运行会被跳过
DATA_DIR=/srv/data/nrel bash run_nrel_v2.sh
```

脚本匹配站点 101275、102811、10444、110192 的 `NREL_WTK_site<站点>_*.csv`，目录中每站应仅有一个匹配文件。默认单 GPU 顺序运行，每次脚本启动先执行服务器检查。可设置 `DEVICE=cuda:1`、`PY=/path/to/python`、`OUT_ROOT=/path/to/new_runs`。中断的非空目录不会被覆盖或自动续训；保留该目录并换一个输出根目录后重跑。

单 CSV 或其他数据集可直接使用通用入口：

```bash
python run_forecast_study.py --config configs/nrel_v2.json \
  --data /srv/data/NREL_WTK_site101275_your_file.csv \
  --horizons 24 96 244 --seeds 42 43 44 --widths 64 \
  --variants baseline residual near residual_near \
  --output-root runs/forecast_v2 --device cuda:0 --skip-complete

python compare_runs.py --study --root runs/forecast_v2 \
  --output runs/forecast_v2/comparison
```

普通 CSV 使用对应的 JSON 配置，例如 `configs/baseline.json`；不要把 NREL 的 `power` 列、划分和输出范围套用到其他数据。

## 3. 输出范围与数据版本

`configs/nrel_v2.json` 设置 `output_constraint=bounded`、`output_min=0`、`output_max=16`，范围使用**原始单位**。这对应上一批文件标注的 16 MW 数据；运行前核对实际单位和额定容量。不同变量需要不同范围时，本版本的统一标量上下界不适用，应关闭限制或分开建模。通用默认配置是 `none`。

- `model` 始终是原始网络预测；`model_bounded` 是同一预测经过范围裁剪的附加结果。裁剪不参与训练损失或最佳检查点选择。
- `range_audit.csv` 统计完整评估集全部预测实例的最小/最大值和越界比例，包含没有观测真值的位置；MSE 仍只对有观测真值的位置评分。
- 同时统计观测真值越界次数，用于发现范围设定不适用的问题。统计容差默认 `1e-5` 原始单位。
- `predict.py` 在限制开启时输出裁剪后的 `forecast.csv` 及未裁剪的 `forecast_raw.csv`。
- 新运行记录原始 CSV 的 SHA-256 和实际首尾时间、间隔统计。重新评估新检查点时，文件内容不同就报错；新预测输入不要求与训练文件相同。

旧模型按默认直接预测模式加载，模型权重结构保持兼容；实际旧检查点加载仍需在服务器验证。旧检查点没有文件指纹，只能使用原有的结构与边界校验，无法倒推证明其训练数据版本。

无需重训即可给服务器保留的旧 `best.pt` 补齐第一步/近端/范围报告：

```bash
python evaluate.py --run-dir /srv/runs/old_run \
  --data /srv/data/the_same_training_file.csv --split val \
  --output runs/old_run_val_audit --device cuda:0 --diagnostics \
  --output-constraint bounded --output-min 0 --output-max 16 \
  --probe-windows 256
```

`best.pt` 通常未上传仓库，旧结果目录中的 JSON/图片不能替代模型文件。重新评估输出放到新目录，保留旧实验结果。

## 4. 编码宽度对照

先只改变 D，比较 D=8/16/32/64 是否影响预测效果：

```bash
python run_forecast_study.py --config configs/nrel_v2.json \
  --data /srv/data/NREL_WTK_site101275_your_file.csv \
  --variants baseline --widths 8 16 32 64 \
  --horizons 24 96 244 --seeds 42 43 44 \
  --output-root runs/width_study --device cuda:0 --skip-complete

python compare_runs.py --study --reference-width 64 \
  --root runs/width_study --output runs/width_study/comparison
```

对照报告记录参数量和训练耗时。耗时仅包含实际训练轮次的训练循环，不含验证、诊断、绘图；早停轮数和服务器负载也会影响它，不是严格硬件性能基准。

D 是每个时间位置的特征宽度，历史位置数仍为 L。这是固定宽度消融，**尚未实现自适应 token 数或样本级动态压缩**。若更小 D 在多个种子上保持验证表现，才能据此讨论减少冗余；不能直接把有效秩当作应保留的 token 数。

## 5. 残差模式的表征诊断

残差预测可以直接使用最后观测值，因此需要区分两条信息路径：

| 干预 | 编码 | 最后值上下文 |
|---|---|---|
| normal | 原编码 | 原上下文 |
| zero / probe_mean / sample_shuffle / time_shuffle | 对应编码干预 | 原上下文 |
| persistence_only | 移除全部学习修正量，包括预测头偏置 | 原上下文 |
| context_shuffle | 原编码 | 其他样本的上下文 |
| joint_shuffle | 其他样本的编码 | 同一个其他样本的上下文 |

`zero` 仍有预测头偏置，不能等同于纯最后值预测。残差模型对编码干预不敏感，也可能只是主要依赖最后值，不能直接判定表征坍缩。图中同时显示原始与跨样本中心化后的窗口均值余弦相似度；中心化相似度同样不应单独作为成功标准。

导出残差模型的 `latent.npy` 时，还需保留 `residual_context.json` 才能复现完整预测；仅 Z 不包含这条显式上下文。

固定一个已训练模型，重复抽样验证探针，检查小样本诊断是否偏离完整验证集：

```bash
python audit_probes.py --run-dir /srv/runs/a_completed_run \
  --data /srv/data/the_same_training_file.csv \
  --output runs/probe_audit --windows 256 --repeats 5 --device cuda:0
```

该脚本不训练，只读取验证集；生成 `audit.html`、`audit.png`、每次诊断 JSON 和汇总 CSV。抽样来自满足间隔要求的候选窗口；同一次中不重复，不同次可以重叠。报告的波动是探针抽样波动，不是多种子训练稳定性，也不是置信区间。若候选池小于请求数，各次可能完全相同，报告会提示，此时应减小 `--windows` 后再分析抽样差异。

## 6. 结果重点看什么

运行目录结构是 `输出根目录/数据名/h预测长度/d宽度/seed种子/组名/`。逐轮报告新增近端/远端曲线，最终报告新增第一步/近端/远端对照、相对最后值的逐步误差比、完整预测越界统计和裁剪前后曲线。

批量报告位于 `comparison/comparison.html`，附带 `runs.csv`、`grouped.csv`、`paired_deltas.csv` 和 `comparison.json`。配对差值为“实验组减基线”，负数代表对应误差降低。模式/损失对照匹配同宽度、同种子基线；纯宽度对照匹配同种子的 D=64 基线。缺少匹配基线就不生成该项配对。误差条为种子间样本标准差。

优先按原始验证 MSE 和第一步/近端/远端误差分析，再看最终测试结果是否一致。代码还没有新的实测结果，不能预先宣称残差、加权或较小 D 一定有效。

新输出默认被 Git 忽略，不会自动推送数据、模型或实验产物。需要后续分析时，将完成运行的配置、汇总指标、训练曲线和诊断报告单独带回即可。
