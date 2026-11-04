# PhysT2V-Bench 提示词与元数据发布说明

[English](README.md) | [简体中文](README_zh-CN.md)

本目录包含 PhysEval / PhysT2V-Bench 流水线所使用的提示词表和评测元数据。

数据集的设计原则是让每个生成视频对应一个可从跟踪运动中测量的主要物理量，每个样本包含单一目标值和对应的物理评估器。

## 文件

- `phys_t2v_bench_prompts.xlsx`：原始 Excel 提示词工作簿。
- `phys_t2v_bench_metadata.xlsx`：精简后的 Excel 评测元数据工作簿，500 行、18 列。
- `csv/`：两个工作簿中每个工作表的 UTF-8 CSV 导出文件，便于检查、版本管理以及不依赖 Excel 的工作流程。

## 数据集规模

本次发布包含 500 组提示词—元数据对：共 10 项物理指标，每项指标 50 个实例。

| 模块 | 指标 | 提示词 ID 前缀 | 数量 | 备注 |
|---|---|---:|---:|---|
| B1.1 | velocity | V | 50 |  |
| B1.2 | acceleration | A | 50 |  |
| B2 | gravity | G | 50 | 仅地球重力 |
| B3.1 | friction_coefficient | F | 50 |  |
| B3.2 | restitution_coefficient | R | 50 |  |
| B3.3 | density_from_initial_acceleration | D | 50 |  |
| B3.4 | fluid_viscosity | ETA | 50 |  |
| B3.5 | spring_constant | K | 50 |  |
| B4.1 | mechanical_energy_conservation | E | 50 |  |
| B4.2 | momentum_conservation_1d | P | 50 |  |

## 任务分类与测量方法

| 类别 | 指标 | 预期物体数 | 测量量 | 空间标定 |
|---|---|---:|---|---|
| 运动学 | `velocity` | 1 | 线性轨迹斜率 | 公制单位需要 |
| 运动学 | `acceleration` | 1 | 二次轨迹系数 | 公制单位需要 |
| 物理常数 | `gravity` | 1 | 自由落体加速度 | 公制单位需要 |
| 材料参数 | `friction_coefficient` | 1 | 减速度与重力加速度之比 | 需要 |
| 材料参数 | `restitution_coefficient` | 1 | 碰撞后/碰撞前速度比 | 不需要；尺度相消 |
| 材料参数 | `density_from_initial_acceleration` | 1 | 流体中早期下落加速度 | 需要 |
| 材料参数 | `fluid_viscosity` | 1 | 结合已知半径和密度的终端速度 | 需要 |
| 材料参数 | `spring_constant` | 1 | 结合已知质量的振荡周期 | 不需要；仅依赖周期 |
| 守恒 | `mechanical_energy_conservation` | 1 | 势能—动能比 | 需要 |
| 守恒 | `momentum_conservation_1d` | 2 | 碰撞前后相对动量误差 | 共用尺度时不需要 |

提示词面向短时单目视频，要求相机固定、运动近似发生在与图像平面平行的平面内、目标边界清晰可见，并尽量减少杂乱背景和遮挡。评测流水线通过跟踪质量门限、物理有效性门限以及连续掩膜可靠性评分，检验并评估生成视频对这些假设的遵循程度。

## 提示词工作簿

`phys_t2v_bench_prompts.xlsx` 包含：

- `All_Prompts`：500 条文本生成视频提示词。
- `Summary`：各模块的提示词数量和 ID 范围。
- `README`：数据来源和匹配键说明。

`All_Prompts` 中的主要列：

- `Global_Index`：全局行号。
- `Module`：基准测试模块。
- `Index`：用于匹配元数据工作簿的索引。
- `Prompt_ID`：提示词标识符和推荐的文件名前缀。
- `English_Prompt`：用于生成视频的提示词文本。

## 元数据工作簿

`phys_t2v_bench_metadata.xlsx` 包含：

- `metadata`：批量评测脚本读取的 500 行评测数据。
- `summary`：各模块的指标数量。
- `field_guide`：字段说明。

`metadata` 按样本标识和三类任务信息排列，单行表头与原字段名保持可供程序读取：

| 分组 | 字段 |
|---|---|
| 样本标识 | `Index`、`Prompt_ID`、`Video_File`、`Tracking_JSON`、`Module` |
| 场景设置 Scene Setup | `Tracking_Objects`、`Num_Objects`、`View`、`Motion_Axis`、`Known_Parameters_JSON` |
| 物理目标 Physical Target | `Metric`、`Target_Type`、`Target_Value`、`Target_Unit` |
| 尺度标定 Scale Calibration | `Scale_Mode`、`Calibration_Object`、`Calibration_Dimension`、`Calibration_Value_m` |

`Metric` 对应的评估函数和测量方法由 [任务定义](../../physics_eval/tasks.py) 提供，主表省去重复的 `Evaluator`、`Measurement_Method` 两列。加载时恢复这两个内部字段，结果文件继续记录它们。旧版 20 列表仍可使用，显式填写的字段（包括空值和自定义值）保持原行为；只有省略的列使用默认值。精简表需搭配新版代码，旧代码可继续使用旧版表。

场景设置是字段分组，不是将数值合并为一段文本。元数据保留评测所需的场景信息，并非完整保存提示词中的颜色、背景等所有描述。具体而言：

- `Num_Objects` 属于跟踪质量硬门限；动量任务需要两条持续存在的轨迹，其他当前任务预期一个物体。
- `Scale_Mode` 表明是否用已知尺寸物体提供米/像素标定，或评估器是否采用空间尺度会相消的纯比值/纯周期物理量。
- `Known_Parameters_JSON` 保留已知物理参数、场景条件和已有生成约束，键名和值均不变。
- `Target_Value` 和 `Target_Unit` 定义用于计算相对误差、可审计的数值目标。

需要空间标定时，提示词中的标准物体或明确尺寸应与元数据中的标定依据对应。比值和周期任务仍遵循原有规则。目标参考值不一定显式写入提示词，例如地球重力任务仍不写入数值 g。

字段精简不改变提示词、物理估计、跟踪、质量门控或评分规则。新旧区别与验证范围见 [兼容性说明](../../docs/METADATA_COMPATIBILITY.md)。

## 使用方法

使用 `phys_t2v_bench_prompts.xlsx` 中的提示词生成视频，并按照对应的 `Video_File` 列命名每个视频。评测程序根据元数据中的 `Video_File` 列精确匹配视频文件名。将生成的视频放置在特定模型的目录下：

```text
data/t2v_videos/<model_name>/
```

关于运行批量评测、目标跟踪和汇总评分的完整指令，请参见[批量使用说明](../../BATCH_USAGE.md)和[评分说明](../../SCORING_USAGE.md)。

## 验证

发布前，已对文件进行以下检查：

- 包含 500 行提示词和 500 行元数据。
- `Index` 和 `Prompt_ID` 键唯一。
- `Known_Parameters_JSON` 中的 JSON 有效。
- 提示词或元数据单元格中不存在本地绝对路径字符串。
