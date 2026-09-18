# Forecast Lab：先预测，再诊断表征

这是一个可以独立放到服务器运行的 PyTorch 项目。目标是回答：**直接监督预测是否足够？编码是否被预测头使用？辅助约束是否确实改善预测？**

交付状态：代码已编写；按要求**未在本机运行 Python、训练、测试或图表渲染**。以下服务器检查命令是交给使用者执行的，不代表已通过。项目没有预置或伪造任何实验分数。

## 实验顺序

1. **直接预测基线**：历史 → 可训练编码器 → 预测头，仅优化未来数值的 MSE。
2. **监测表征**：每轮在固定验证探针上统计方差、有效秩、相关性和相似度。
3. **编码干预**：冻结参数，将编码清零、替换为均值、跨样本打乱或打乱时间位置，比较同一批目标上的误差。
4. **按证据做对照**：显式开启方差约束、协方差约束或辅助重构；默认都关闭。不根据低秩自动开启正则，不根据测试指标选择权重。

默认模型没有 MoE、自适应压缩或必须可逆的编码约束。仓库名称不意味着基线需要引入 MoE。模型是本项目实现的 TCN 基线，不是 PatchTST / TS2Vec / VICReg 的论文复现。

## 服务器安装

建议使用 Python 3.10–3.12 的独立环境。命令以 Linux 服务器为例：

```bash
git clone https://github.com/Aersd1/token_and_moe.git
cd token_and_moe/forecast_lab
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

GPU 服务器先根据驱动和 CUDA 环境安装对应的 PyTorch，使用 [PyTorch 官方安装选择器](https://pytorch.org/get-started/locally/)。随后：

```bash
python -m pip install -r requirements.txt
```

这是兼容版本范围，不是已经在本机验证的环境锁文件。固定服务器环境后可自行保存 `pip freeze`。默认单进程、单 GPU；不包含 DDP。

## 数据格式与路径

CSV 格式与 ETT、weather、electricity、traffic、exchange_rate、illness 相同：

```text
date,feature_1,feature_2,OT
2016-07-01 00:00:00,...,...,...
2016-07-01 01:00:00,...,...,...
```

以上省略号仅说明列结构，不是可输入的数值。真实数据留在服务器自己的数据目录中，仓库不会上传本地 CSV。

默认行为：

- 自动识别名为 `date/time/timestamp/datetime` 的时间列，其余列必须为数值。
- 所有数值列作为输入和预测目标（多变量 → 多变量）。使用 `--targets OT` 可预测单个目标，同时保留其他列作为输入。
- `--features 'HUFL,HULL,MUFL,MULL,LUFL,LULL,OT'` 指定输入列；目标必须是输入列的子集。
- 时间列必须有序、唯一；不会静默排序或删除行。不规则间隔会提示，预测长度仍按**行数**计算。
- 无时间列时使用 `--time-column none`，并确保选中的输入列都为数值。
- 默认 UTF-8。必要时明确指定 `--encoding gb18030` 或文件实际编码，不做静默编码猜测。

默认按行数划分 70% / 10% / 20%，分别作为训练、验证和测试。`--split ett-hour` 使用 ETT 常见的 12/4/4 个 30 天月划分；`--split ett-minute` 在此基础上乘以 4。ETT 模式不足 20 个月会报错，多出的尾部行不使用，实际边界写入报告。**ratio 模式结果不能直接冒充论文固定 ETT 划分结果。**

每个窗口的整个未来目标都在所属区间内；验证和测试的历史可以使用之前已经观测到的数据。这是滚动预测协议，不是仅在测试区间起点做一次整段递推。默认每行一个预测起点，因此不同起点的目标可能重叠；指标统计的是“起点 × 预测步 × 变量”的预测实例。`--eval-stride` 可控制评估起点间隔。

标准化只拟合训练区间，每列分别计算均值和标准差。默认缺失值/inf 报错；`--missing ffill` 仅用过去值前向填充，开头缺失使用训练均值，**不进行反向填充或双向插值**。缺失的目标值通过掩码排除，不以填充值作为训练/评分真值。即使某段无缺失，生产场景的长时间断流仍需要使用者单独处理。

## 第一步：服务器完整性检查与最小运行

以下命令只在目标服务器执行：

```bash
python server_checks.py

python train.py \
  --data /srv/data/ETT-small/ETTh1.csv \
  --lookback 32 --horizon 8 --d-model 16 --layers 2 \
  --epochs 2 --batch-size 16 --probe-windows 8 \
  --output runs/server_smoke --device cuda
```

`server_checks.py` 使用临时合成数据检查：修改验证/测试数值不能改变训练标准化；预测目标不跨区间；前向填充不读取未来；位置编码本身不能掩盖跨样本坍缩；预测损失能反传到编码器。它不是性能基准。第二条命令使用真实 CSV 检查训练、保存、评估及报告全流程；两轮结果不用于科学结论。

没有 GPU 时将 `--device cuda` 改为 `--device cpu`。输出目录必须新建或为空，避免覆盖已有实验。

## 第二步：直接预测基线 + 监测 + 干预

```bash
python train.py \
  --config configs/baseline.json \
  --data /srv/data/ETT-small/ETTh1.csv \
  --split ett-hour \
  --output runs/etth1_baseline \
  --device cuda --tensorboard
```

参数优先级：命令行 > JSON 配置 > 默认值。JSON 内的键使用下划线，例如 `variance_weight`；命令行使用连字符，例如 `--variance-weight`。

仅预测 OT：

```bash
python train.py --data /srv/data/ETT-small/ETTh1.csv \
  --targets OT --split ett-hour --output runs/etth1_ot --device cuda
```

小规模 illness 数据使用较短的历史和预测区间：

```bash
python train.py --config configs/illness.json \
  --data /srv/data/illness/national_illness.csv \
  --output runs/illness_baseline --device cuda
```

服务器长时间运行：

```bash
nohup python -u train.py --config configs/baseline.json \
  --data /srv/data/weather/weather.csv \
  --output runs/weather_baseline --device cuda \
  > weather_console.log 2>&1 &
```

### 模型结构

```text
历史 X [B, L, C]
  → 逐时间点线性嵌入 + 固定位置编码
  → 多层残差空洞卷积 + LayerNorm
  → Z [B, L, D]
  → 时间维线性映射 L→H + 目标维映射 D→C_target
  → 未来预测 [B, H, C_target]
```

- 时间长度始终为 L，没有 token 删除、池化或自适应长度选择。
- D 是特征宽度，不是 token 数。C→D 的嵌入可能降维；本项目没有宣称表示无损或完全不发生信息压缩。
- 卷积可查看完整的已观测历史窗口，无法访问窗口后的真实未来。适合窗口级预测，不是逐点在线缓存的因果编码器。
- 预测头只接收 Z，没有原始 X 到预测头的旁路；时间位置和预测头偏置仍可表达不依赖输入的基线。
- 预测头采用分离的时间/变量映射，避免 `L*D → H*C` 的巨大全连接矩阵。

### 监测的含义

探针在训练前固定，默认来自验证集。默认相邻探针起点至少间隔 H 行，目标区间不重叠，但历史可能重叠。如需历史和目标均不重叠，设置 `--probe-spacing L加H的数值`（例如 L=96、H=96 时填 192）。这会减少样本数量；不足两个样本时秩和方差诊断标为不可用，不伪造数值。

对选中的 P 个时间位置，先跨 N 个样本中心化，再计算：

```text
C = Σ_t Σ_i (z_it - mean_i(z_it)) (z_it - mean_i(z_it))ᵀ / [P (N-1)]
```

这排除了“样本编码全相同，但不同时间位置编码不同”的假象。报告包含：

| 指标/图表 | 含义与限制 |
|---|---|
| Cross-window std | 同时间位置跨样本变化，按位置汇总到每个特征 |
| Near-zero fraction | 方差很小的维度比例；阈值与尺度有关，不自动判坍缩 |
| Effective rank | 协方差特征值归一化后的熵指数；显示样本量决定的秩上限 |
| Pooled effective rank / PCA | 窗口时间均值的统计，可能遗漏动态信息，单独标明 |
| Feature correlation | 匹配时间位置的跨样本协方差归一化结果 |
| Window-mean cosine | 窗口均值之间的相似性，可能受公共分量影响 |
| Adjacent token cosine | 相邻时间位置相似性；只是过度平滑线索 |

默认 `--probe-windows 128 --probe-positions 16`，监测次数和规模可控。有效秩以协方差特征值为基础，不是矩阵奇异值熵的另一种口径。LayerNorm 不会保证这些跨样本维度方差非零。

### 编码干预

全部在 `eval()` / 无梯度下运行，使用同一批探针和相同目标：

| 模式 | 操作 |
|---|---|
| normal | 原编码 |
| zero | 所有编码清零 |
| probe_mean | 每个位置替换为探针样本在该位置的均值 |
| sample_shuffle | 将完整编码分配给其他样本，保证没有样本拿到自己的编码 |
| time_shuffle | 保留样本归属，打乱其编码时间位置 |

报告保存标准化和原始量纲误差，以及相对 normal 的绝对/相对变化。`probe_mean` 依赖整批探针输入，**只用于诊断，不能当成可部署预测器或正式基线**。打乱是一次固定种子的干预，不代表重复抽样的置信区间。干预可能产生分布偏移；“误差增加”说明模型对这种扰动敏感，不足以单独证明表征因果重要性。

## 第三步：按证据启用约束

总损失：

```text
L = L_prediction + λ_rec L_reconstruction + λ_var L_variance + λ_cov L_covariance
```

- `L_prediction`：有观测目标上的标准化 MSE。
- `L_reconstruction`：从 Z 通过辅助线性头重构历史，只对实际观测的历史评分。
- `L_variance`：跨样本、匹配时间位置的每维标准差低于目标值时惩罚。
- `L_covariance`：匹配时间位置的跨样本协方差，惩罚非对角项。

方差/协方差项借鉴 VICReg 的统计正则思路，没有增强视图匹配项，**不是完整 VICReg**。最后一个 batch 只有一个样本时跳过这两个正则。给出的权重只是起始对照值，没有在本地调优或验证有效。

```bash
# 方差约束
python train.py --config configs/baseline.json \
  --data /srv/data/ETT-small/ETTh1.csv --split ett-hour \
  --variance-weight 0.01 --output runs/etth1_variance --device cuda

# 方差 + 去相关
python train.py --config configs/baseline.json \
  --data /srv/data/ETT-small/ETTh1.csv --split ett-hour \
  --variance-weight 0.01 --covariance-weight 0.001 \
  --output runs/etth1_varcov --device cuda

# 单独验证辅助重构是否有用
python train.py --config configs/baseline.json \
  --data /srv/data/ETT-small/ETTh1.csv --split ett-hour \
  --reconstruction-weight 0.1 --output runs/etth1_reconstruction --device cuda
```

应固定数据划分、历史长度、预测长度、模型、训练预算和种子集合，再比较。重构可能保留噪声，维度去相关也可能破坏有用的相关结构；不保证这些约束一定有帮助。

## 批量实验：多个数据集、预测长度与种子

保持当前数据目录结构即可：

```bash
# 第一阶段仅运行基线，报告中已自动包含监测和编码干预
python run_suite.py --data-root /srv/data \
  --datasets ETTh1 weather exchange_rate \
  --base-config configs/baseline.json \
  --horizons 96 192 --seeds 42 43 44 \
  --stage baseline --output-root runs/study --device cuda

# 看过基线报告后，使用完全相同的协议运行正则对照
python run_suite.py --data-root /srv/data \
  --datasets ETTh1 weather exchange_rate \
  --base-config configs/baseline.json \
  --horizons 96 192 --seeds 42 43 44 \
  --stage regularized --variants variance varcov reconstruction \
  --output-root runs/study --device cuda

python compare_runs.py --root runs/study --output runs/study/comparison
```

这些示例全部使用 ratio 划分。需要 ETT 固定划分时，为 ETT 单独创建 JSON（设置 `"split": "ett-hour"` 或 `"ett-minute"`）并单独运行；不要将 ETT 专用划分用于 weather/illness。illness 批量运行需显式使用 `--base-config configs/illness.json --horizons 24 36`，套件的默认 horizon 为 96。

支持的名称：ETTh1、ETTh2、ETTm1、ETTm2、weather、electricity、traffic、exchange_rate、illness。任意其他文件使用 `--data /path/custom.csv`。

`--stage regularized` 要求同一输出树中已有对应基线，且非正则实验配置一致。`--stage all` 是显式一次性运行预先选定的全部对照，不代表程序判断出了坍缩。套件顺序运行，不会自动占用多张 GPU。`--skip-complete` 只跳过配置一致的已完成运行，**不续训半途终止的运行**；后者需使用新的输出目录。

对照报告按验证误差排序，测试误差仅作最终报告。只对相同协议和相同权重的不同种子汇总均值、样本标准差；单种子不显示误差条。`paired_deltas.csv` 按相同种子匹配基线，差值为“变体−基线”。不要根据多次查看测试集来选择最佳方法。

## 图表和输出

```text
runs/etth1_baseline/
  config.json                  完整训练配置
  data_metadata.json           列名、缩放参数、区间边界、缺失统计
  provenance.json              Python/PyTorch/CUDA/设备/Git 提交
  train.log                    训练日志
  best.pt                      验证 MSE 最小的模型与预处理参数
  history.csv                  逐轮预测/辅助损失、学习率
  monitor_history.csv          逐轮表征统计和干预误差
  probe_origins.json           固定探针起点
  diagnostics/epoch_0000.json   实际初始化模型的诊断
  diagnostics/epoch_XXXX.json   对应轮次诊断
  diagnostics/best_validation.json
  validation/                  最佳检查点的完整验证指标与示例
  test/                        最佳检查点的完整测试指标与示例
  summary.json                 对照汇总入口
  report_state.json            可视化使用的原始记录
  report.html                  自包含交互报告，无需联网
  figures/*.png                完成训练后导出的静态图
  tensorboard/                 显式开启后生成
```

报告会在每轮刷新，包含学习曲线、表征轨迹、维度活跃度、特征谱、相关性热图、PCA、样本相似度、干预误差、预测基线比较、不同预测步误差、变量误差和原始量纲预测曲线。最终报告的诊断和预测来自**同一个最佳检查点**；不是最后一轮权重。初始化诊断的 epoch=0 是真实运行结果，不是零值占位。

HTML 内嵌 Plotly，不依赖 CDN，可从服务器下载到本机打开；支持缩放、悬停、图例筛选和 PNG 下载。静态 PNG 由 Matplotlib 生成，不需要浏览器或 Kaleido。HTML 每轮更新，PNG 默认训练结束时导出；训练中也可手动导出已有监测：

```bash
python render_report.py --run-dir runs/etth1_baseline --png
tensorboard --logdir runs/etth1_baseline/tensorboard --host 127.0.0.1 --port 6006
```

TensorBoard 仅在训练时使用了 `--tensorboard` 才有数据。远程访问可使用 SSH 端口转发。训练失败时保存 `failure.json` 与可读取的已有报告，失败目录不算已完成实验。

## 重新评估与实际预测

冻结最佳检查点，对原数据重新评估；数据搬家时通过 `--data` 指向相同 CSV：

```bash
python evaluate.py --run-dir runs/etth1_baseline \
  --data /srv/data/ETT-small/ETTh1.csv \
  --split test --output runs/etth1_test_review \
  --device cuda --diagnostics
```

重新评估重用训练缩放参数，不重新拟合。检查列结构、划分长度及首尾时间；**这些检查不等同于逐字节文件校验**，请保留原始数据版本。测试集诊断仅用于事后分析，不用于调参。

使用训练好的编码器，对新 CSV 最后的历史窗口预测未来，不需要提供未来标签：

```bash
python predict.py --checkpoint runs/etth1_baseline/best.pt \
  --input /srv/data/new_history.csv \
  --output runs/new_forecast --device cuda --save-latent
```

新数据需要相同输入列和至少 L 行历史。输出 `forecast.csv`；`--save-latent` 额外导出形状 `[L,D]` 的 `latent.npy`，可供后续表征研究使用。仅当输入时间间隔恒定时推算未来时间戳，否则只提供预测步序号。所有预测仍使用训练时的单位与缩放参数。

## 资源和评估边界

- 对 traffic/electricity 等高维数据，显存不足先减小 `--batch-size` 与 `--diag-batch-size`，再考虑减小 D；参数变化后应重新运行匹配基线。
- `--monitor-every 5 --probe-windows 64 --probe-positions 8` 可降低监测开销。完整验证仍每轮进行，测试只在选定最佳检查点后进行。
- 可选 `--amp` 使用 CUDA 混合精度；默认 float32。`--deterministic` 启用严格确定性，不支持确定性实现的算子会报错。
- 学习率下降和早停只读取验证 MSE。`--min-delta` 只控制早停耐心，保存的 `best.pt` 始终对应最低验证 MSE。
- MAE/MSE 同时报告标准化与原始量纲。跨单位变量的原始 MSE 可被大数值变量主导，应结合逐变量结果。
- 正式无训练基线为 last_value 和 history_mean；指定 `--seasonal-period 24` 可加入季节重复基线，周期需符合数据采样频率且不超过历史长度。
- 当前不包含断点续训、概率预测、外生未来已知变量、概念漂移处理或自动超参数搜索。
- 低秩、高余弦相似度、平滑预测均不是自动失败判据。应结合验证效果、朴素基线与编码干预共同判断。

## 参考思路

- [PatchTST](https://arxiv.org/abs/2211.14730)：直接监督预测与自监督预训练是可分开的实验路线。
- [VICReg](https://arxiv.org/abs/2105.04906)：方差与协方差正则；本项目仅借鉴统计约束，并按时序位置处理。
- [TLAE](https://ojs.aaai.org/index.php/AAAI/article/view/17101)：预测与重构联合目标的时序研究。

以上引用用于说明设计来源，不构成复现或性能优于论文方法的声明。
