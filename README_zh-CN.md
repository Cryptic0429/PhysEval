# PhysEval

[English](README.md) | [简体中文](README_zh-CN.md)

**量化视频生成结果与物理规律之间的差距。**

PhysEval 是用于评估文生视频结果是否满足**定量物理约束**的基准数据集和自动评测流程。它关注的不只是画面看起来是否合理：当提示词指定一个物理量时，生成视频中能否测出对应运动，以及测量值是否接近目标值？

每条基准样本包含提示词和可核查的评测元数据：物理指标、目标值及单位、评估器、预期物体数、尺度设置和已知物理参数。给定生成视频后，流程会初始化并跟踪物体，检查视频是否可测量，估计相应物理量，并输出归一化分数与失败原因。

## 基准概览

- **500 组提示词与元数据**：10 项物理指标，每项 50 组。
- **四类任务**：运动学、物理常量、材料参数和守恒类任务。
- **结果可追溯**：每条分数可追溯到提示词、目标、轨迹、测量值、有效状态、掩膜质量和剔除原因。
- **面向文生视频**：提示词要求固定机位、近似平面运动、清晰可见的目标，以及便于自动测量的简单场景。

| 类别 | 指标 | 从视频估计的量 | 预期物体数 |
| --- | --- | --- | ---: |
| 运动学 | 速度 | 线性轨迹斜率 | 1 |
| 运动学 | 加速度 | 二次轨迹系数 | 1 |
| 物理常量 | 重力加速度 | 自由落体加速度 | 1 |
| 材料参数 | 摩擦系数 | 减速比 | 1 |
| 材料参数 | 恢复系数 | 碰撞后与碰撞前的速度比 | 1 |
| 材料参数 | 密度 | 流体中下落的初始加速度 | 1 |
| 材料参数 | 流体黏度 | 终端速度 | 1 |
| 材料参数 | 弹簧劲度系数 | 振动周期 | 1 |
| 守恒 | 机械能守恒 | 能量比 | 1 |
| 守恒 | 一维动量守恒 | 相对动量误差 | 2 |

发布的提示词和元数据工作簿见[基准元数据说明](benchmark/metadata/README_zh-CN.md)。

## 评测流程

PhysEval 将**视频是否可测量**与**物理量是否准确**分别处理：

1. **物体跟踪**：借助 YOLO 初始化和 SAM2 掩膜传播，提取可见物体的轨迹。
2. **跟踪质量门控**：检查预期物体数、有效帧覆盖率和有效运动；未通过的视频不进入有效评分集合。
3. **物理估计有效性门控**：各指标评估器返回 `valid`、`weak_valid`、`invalid` 或 `failed`；只有前两类进入有效集合。
4. **掩膜可靠性**：通过硬门控后，取值在 `[0, 1]` 的掩膜质量分作为二维轨迹可靠性的软指标，降低分数，但在论文默认口径下不会单独剔除视频。
5. **物理评分**：将相对误差换算为按指标归一化的物理分数，再结合跟踪、掩膜质量及估计有效性因素。

论文分别报告三种结果：

- **有效视频得分（主要的物理准确性结果）**：通过两道硬门控的视频，其调整后得分的均值。
- **剔除率（可测量性结果）**：因跟踪质量或物理估计有效性不足而被剔除的视频比例。
- **端到端得分（补充结果）**：以全部视频为分母，给被剔除的视频记零分。它同时反映可测量性和物理准确性，不能代替前两项结果。

公式、阈值、状态规则和输出字段见[评分说明](SCORING_USAGE.md)。

## 适用范围与假设

当前评估器使用简明、可检查的逆物理模型。它们通常假设视频较短、相机固定、运动近似平行于图像平面、时间基准已知或可推断、二维轨迹稳定，并且在需要物理单位时能从基准元数据取得尺度信息。

本项目不能证明视频满足完整的三维物理规律。深度漂移、相机运动、透视变化、遮挡、形变、复杂流体、旋转或长时间交互都可能使二维估计失效。跟踪门控和掩膜质量可揭示一部分问题，但不能证明场景确实是平面运动。因此，基准得分应作为诊断性测量，而非高风险仿真的正确性认证。

## 仓库内容

- `benchmark/metadata/`：提示词工作簿、评测元数据和 UTF-8 CSV 导出。
- `physics_eval/`：各任务的物理评估器和批量评测逻辑。
- `scripts/`：SAM2 跟踪、YOLO 初始化、掩膜检查、批处理和评分脚本。
- `physeval.py`：统一入口，提供 `eval`、`score` 及可选的 `compare`、`compare-summary` 命令。
- [`compare/`](compare/README.md)：可选的检测器与跟踪器比较模块及已保存的报告。
- `requirements-compare.txt`：比较模块的可选依赖，同时包含基础依赖。
- [`BATCH_USAGE.md`](BATCH_USAGE.md)：批量运行及故障排查说明。
- [`SCORING_USAGE.md`](SCORING_USAGE.md)：评分口径及报告解读。
- `configs/scoring/`：评分程序使用的版本化掩膜质量规则。

代码仓库不包含生成视频、SAM2 源码及检查点、YOLO 权重或批量运行结果。运行评测时需要在本地提供视频。

## 安装

先安装 Python 依赖。GPU 服务器上的 `torch` 和 `torchvision` 需要与 CUDA 环境匹配；若已有可用的 PyTorch 环境，可保留该环境并安装其余依赖。

```bash
pip install -r requirements.txt
```

准备外部 SAM2 和 YOLO 资源：

```bash
bash scripts/setup_sam2_assets.sh
```

默认情况下，此脚本会：

- 将 `https://github.com/facebookresearch/sam2.git` 克隆到 `repo/sam2/`；
- 下载 `sam2.1_hiera_base_plus.pt` 到 `repo/sam2/checkpoints/`；
- 下载 `yolov8n.pt` 到仓库根目录。

其他用法：

```bash
# 使用更大的 SAM2 检查点。
bash scripts/setup_sam2_assets.sh --sam2-model large

# 准备资源并以可编辑模式安装 SAM2。
bash scripts/setup_sam2_assets.sh --install-sam2

# 只使用运动初始化时跳过 YOLO。
bash scripts/setup_sam2_assets.sh --skip-yolo
```

如果准备资源时没有使用 `--install-sam2`，还需安装 SAM2：

```bash
pip install -e repo/sam2
```

## 快速开始

以下命令均从仓库根目录运行。按 `benchmark/metadata/phys_t2v_bench_prompts.xlsx` 中的提示词生成视频，依照元数据工作簿的 `Video_File` 列命名，并将同一模型的视频放在：

```text
data/t2v_videos/<model_name>/
```

运行跟踪与物理评测：

```bash
python physeval.py eval \
  --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx \
  --video-root data/t2v_videos/<model_name> \
  --output-dir batch_eval_results/<model_name> \
  --detector yolo_then_motion
```

按论文口径为结果评分：

```bash
python physeval.py score \
  --result-root batch_eval_results/<model_name> \
  --model-name <model_name> \
  --weak-valid-multiplier 0.8
```

`weak_valid` 的命令行默认权重为 `0.8`，与论文口径一致；参数仍可用于显式的敏感性分析。主要输出结构为：

```text
batch_eval_results/<model_name>/score_reports/<protocol_id>/all_metrics/
  score_summary.json
  metric_scores.csv
  per_video_scores.csv
  score_exclusion_reasons.csv
  plots/
```

批量命令、复用模式、输出布局和诊断方法见[批量使用说明](BATCH_USAGE.md)。

### 评分规则与输出

默认方法将掩膜质量证据分为三个等权组，再对所需物体取平均。缺少证据时，该项得分不可用，不会以满分填补。重现旧评分行为时可以显式选择旧规则；具体选项和实现说明见[评分说明](SCORING_USAGE.md)。

逐视频得分结合物理准确性、测量质量和估计有效性：

```text
最终得分 = 物理准确性得分 × 测量质量系数 × 估计有效性系数
```

物理准确性项将相对误差换算为对应指标的分数。跟踪和物理有效性门控决定视频能否进入有效集合；较低但可测得的掩膜质量分本身不会剔除视频。`weak_valid` 默认权重为 `0.8`。报告写入 `score_reports/<protocol_id>/<scope>/`，包含模型和指标汇总、逐视频分数及剔除原因。

## 可选：检测器与跟踪器比较

核心评测与比较模块共用掩膜质量评分，默认采用三个等权质量组。公式、证据完整性检查和离线敏感性分析命令见[评分说明](SCORING_USAGE.md)。

核心 `eval` 和 `score` 命令不会加载比较模块，也不要求安装 TAM/XMem 依赖。研究检测器或跟踪器差异时，可单独安装并运行比较模块：

```bash
python -m pip install -r requirements-compare.txt
bash compare/scripts/setup_compare_assets.sh --install-editable
python physeval.py compare --video-root data/t2v_videos/model_name --model-name model_name --dry-run
# 移除 --dry-run 后执行比较。
python physeval.py compare-summary --model-name model_name
```

比较模块与核心项目共用 `benchmark/metadata/`、`repo/` 和 `yolov8n.pt`。新生成的输出和帧缓存分别位于 `compare/outputs/` 和 `compare/cache/`；已有报告保留在 `compare/results/`。这些报告衡量跟踪和掩膜质量，**不是**物理准确性分数。模式和完整命令见[比较模块说明](compare/README.md)。

原有的 `scripts/run_batch_simple.py`、`scripts/score_results.py` 以及 `compare/scripts/` 入口仍可使用。脚本位置和默认资源路径均相对当前检出目录解析，因此克隆后可以重命名外层目录。
