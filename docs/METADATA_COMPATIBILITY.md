# Metadata 精简与兼容性

本次调整将逐样本重复的执行配置集中到 `physics_eval/tasks.py`，保留原有计算流程。发布表从 20 列减少到 18 列，仍包含 500 个样本。图中的 Scene Setup、Physical Target、Scale Calibration 对应字段分组，不替代程序所需的结构化数值。

## 新旧区别

| 项目 | 旧版 | 新版 |
|---|---|---|
| 主表 | 500 行、20 列 | 500 行、18 列 |
| 评估函数 | 每行填写 `Evaluator` | 由 `Metric` 的任务定义补齐 |
| 测量方法名称 | 每行填写 `Measurement_Method` | 由同一任务定义补齐 |
| 字段组织 | 任务、算法、场景、标定信息交错 | 样本标识、场景设置、物理目标、尺度标定 |
| 已知物理输入 | `Known_Parameters_JSON` 中的结构化数值与条件 | 键和值完全保留 |
| 目标与标定 | 每样本的目标、单位、参考物体和尺寸 | 值完全保留 |
| 结果文件 | 包含 `Evaluator`、`Measurement_Method` | 继续记录同名字段和执行值 |
| 旧版输入 | 显式指定两个执行字段 | 新代码继续接受，显式值仍优先 |
| 提示词 | 500 条原有提示词 | 文件内容不变 |

例如，摩擦任务原本需要同时指定：

```text
Metric              = friction_coefficient
Evaluator           = eval_friction
Measurement_Method  = deceleration_fit
```

新版输入只需保留：

```text
Metric              = friction_coefficient
```

`tasks.py` 中的固定定义补回后两项。每个样本自己的目标值、标定长度和已知参数仍由该行提供。原本重复填写的 1,000 个执行配置单元格改为 10 项任务定义。

## 新版列组织

| 分组 | 字段 |
|---|---|
| 样本标识 | `Index`、`Prompt_ID`、`Video_File`、`Tracking_JSON`、`Module` |
| Scene Setup | `Tracking_Objects`、`Num_Objects`、`View`、`Motion_Axis`、`Known_Parameters_JSON` |
| Physical Target | `Metric`、`Target_Type`、`Target_Value`、`Target_Unit` |
| Scale Calibration | `Scale_Mode`、`Calibration_Object`、`Calibration_Dimension`、`Calibration_Value_m` |

场景设置保存评测所需信息，未增加对提示词全部视觉细节的结构化抽取。标定信息继续由原有提示词和 metadata 配对提供。本次未重新生成提示词、修改尺寸，也未新增运行时的自然语言校验门槛。

## 运行逻辑

```text
旧版：20 列 metadata → 原有跟踪、物理估计、质量检查和评分
新版：18 列 metadata → 补齐两个固定字段 → 同一套跟踪、物理估计、质量检查和评分
```

新增的 `resolve_task_fields()` 返回输入的副本，只补充缺失的键，不改写已有值。Excel 加载、CSV/批处理加载和直接调用 `process_row()` 都支持精简输入。结果结构、CLI 命令、任务标识、目录命名、计算公式、阈值和评分容差保持原样。

为保留旧格式行为，显式填写的值（包括空值、自定义方法标签或未知评估函数）不被默认值覆盖。原有错误仍由原流程处理。未知任务和 tracking-only 模式不套用这十项默认定义。

| 代码与数据组合 | 支持情况 |
|---|---|
| 新代码 + 精简版 18 列表 | 支持，加载时补齐 |
| 新代码 + 原版 20 列表 | 支持，保留显式值 |
| 旧代码 + 原版 20 列表 | 原有行为 |
| 旧代码 + 精简版 18 列表 | 不支持，旧代码没有补齐逻辑 |

新增真实不同的测量方法时，需要明确的任务/方法设计，不能只改变展示名称。本次仅集中当前十个任务已有的固定对应关系。

## 验证结果

比较基线为本次修改前的代码与原始 metadata（仓库提交 `133153c`）。旧、新代码在独立进程中读取各自版本的 XLSX，并处理相同的控制轨迹。

| 检查 | 结果 |
|---|---|
| 500 条样本、18 个保留字段与旧版逐值比较 | 全部一致 |
| 补齐后的 500 条完整 metadata 与旧版比较 | 全部一致 |
| 新版 XLSX 与 CSV 的运行时记录比较 | 全部一致 |
| 500 条正常长度控制轨迹的物理结果与分数 | 完全一致 |
| 500 条帧数不足控制轨迹的失败结果与分数 | 完全一致 |
| 各指标和模型汇总 | 完全一致 |
| 旧版测试 | 97 项测试、14 项子测试通过 |
| 新版测试 | 136 项测试、14 项子测试通过 |

独立回放的 1,000 个案例包含 450 个 `valid`、50 个 `weak_valid` 和 500 个 `failed` 结果。比较完整结果字段，未使用数值容差放宽差异。另有现有测试覆盖 invalid、质量门控、掩膜评分和报告契约。

物理估计器、数学计算、跟踪、质量检查、尺度换算、评分、流水线编排和结果序列化的原有实现未修改，提示词工作簿未修改。此次没有重跑真实视频的 GPU 推理；验证使用控制轨迹、已有回归样例及模拟跟踪测试，不能据此声称重新验证了 YOLO/SAM2 的模型精度。

可重复运行的仓库检查：

```bash
python -m pytest tests
```

专项测试位于 `tests/test_metadata_compatibility.py`。`tests/fixtures/legacy_metadata.csv` 保存每个任务的一条原版记录，作为独立的字段恢复基线。公开流水线回归测试同时覆盖精简和原版输入、两种输出布局、有效视频、缺失跟踪和静止轨迹。
