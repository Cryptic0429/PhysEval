# Metadata 输入格式与兼容性

发布的 `phys_t2v_bench_metadata.xlsx` 包含 500 个样本和 18 个输入列。各指标固定的执行配置集中在 `physics_eval/tasks.py`，每个样本自己的场景、物理目标和标定数值仍由对应行提供。

## 字段分组

| 分组 | 字段 |
|---|---|
| 样本标识 | `Index`、`Prompt_ID`、`Video_File`、`Tracking_JSON`、`Module` |
| Scene Setup | `Tracking_Objects`、`Num_Objects`、`View`、`Motion_Axis`、`Known_Parameters_JSON` |
| Physical Target | `Metric`、`Target_Type`、`Target_Value`、`Target_Unit` |
| Scale Calibration | `Scale_Mode`、`Calibration_Object`、`Calibration_Dimension`、`Calibration_Value_m` |

这些分组用于组织输入字段。提示词与元数据共同提供场景和标定信息，评测程序直接读取表中的结构化数值。

## 任务字段解析

输入表无需重复填写 `Evaluator` 和 `Measurement_Method`。加载时，`resolve_task_fields()` 根据 `Metric` 从任务定义中补齐这两个字段。例如：

```text
Metric              = friction_coefficient
Evaluator           = eval_friction
Measurement_Method  = deceleration_fit
```

其中只需在输入表中提供 `Metric`，后两项由任务定义解析。结果文件继续记录 `Evaluator`、`Measurement_Method` 及实际执行值。

`resolve_task_fields()` 返回输入的副本，只补充缺失的键，不改写已有值。Excel 加载、CSV/批处理加载和直接调用 `process_row()` 均支持这种输入格式。

## 兼容含显式执行字段的输入

包含 `Evaluator` 和 `Measurement_Method` 的 20 列输入仍可直接使用。显式值始终优先，包括空值、自定义方法标签和未知评估函数；只有缺失字段才采用任务默认值。未知任务和 tracking-only 模式不套用这十项任务的默认定义。

如果运行环境不包含 `physics_eval/tasks.py` 中的字段解析逻辑，应使用含显式执行字段的 20 列输入表。

## 相关测试

```bash
python -m pytest tests
python -m pytest tests/test_metadata_compatibility.py
```

`tests/fixtures/legacy_metadata.csv` 为每项任务提供一条显式 20 列记录，用于检查字段恢复、数值估计、评分和显式覆盖行为。流水线回归测试覆盖两种输入格式、两种输出布局，以及有效视频、缺失跟踪和静止轨迹。
