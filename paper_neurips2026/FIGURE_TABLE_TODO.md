# PhysEval NeurIPS 2026 图表与待补内容清单

当前论文已按新大纲重写：

```text
Introduction -> Related Work -> Benchmark Datasets -> Evaluation -> Experiments -> Discussion -> Conclusion
```

其中 `Limitations` 暂时放在 `Experiments / 5.4`，因为它主要解释实验分数的边界和可解释性，不单独作为顶层章节。
第一次 review 修改后，`Related Work`、`\bench{}` 和 `Evaluation Metrics` 已增加 subsection；`Experiment Setup` 改为正文描述，不再保留占位设置表。

## Section 结构

1. **Introduction**
   - 说明视觉真实不等于物理数值正确。
   - 强调从 physical commonsense 到 quantitative physical evaluation 的区别。
   - 待补：最终数据规模、模型数量、最好模型、主要发现。

2. **Related Work**
   - 参考 `2511.19836` 的写法，把相关 benchmark 分成 general video benchmark、physics-aware benchmark、world-generation benchmark。
   - 已加入 checkmark 风格 benchmark 对比表，突出 PhysEval 与已有 benchmark 的区别。

3. **Benchmark Datasets**
   - 参考 `2410.05363` 的 benchmark 描述方式。
   - 介绍任务类别、dataset specification、batch output layout，并把 planar-motion assumption 简洁并入 dataset specification。
   - Planar-motion assumption 正文只保留概述，详细的 \(Q_{\mathrm{mask}}\) 解释放在附录。
   - 待补：最终 metadata 统计。

4. **Evaluation Metrics**
   - 讲物理量误差、tau、weak_valid、tracking gate、mask continuous score。
   - 当前逻辑：tracking/physics validity 决定 effective；mask 只作为 0-1 分数乘子。

5. **Experiments**
   - 5.1 Experiment Setup
   - 5.2 Discard Rate
   - 5.3 Quantitative Evaluation
   - 5.4 Limitations
   - Experiment Setup 参考 `2410.05363` 改为 prose，不使用占位表。

6. **Conclusion**
   - 总结 benchmark 和最终实证发现。

## 必须补的图

1. **Figure 1: Benchmark overview**
   - 位置：Introduction。
   - 内容：metadata/prompt -> T2V model -> generated videos -> YOLO/SAM2 tracking -> trajectory/mask diagnostics -> physical evaluator -> result JSON -> model score。
   - 注意：需要手工精修，不建议直接使用 Python debug 图。

2. **Figure 2: Discard-rate and validity diagnostics**
   - 位置：Experiments / Discard Rate。
   - 内容：per-metric physical-valid rate、tracking-pass rate、effective-video rate、top exclusion reasons。
   - 注意：exclusion reasons 是 multi-label，柱子相加可以大于总视频数。

3. **Figure 3: Scores by metric**
   - 位置：Experiments / Quantitative Evaluation。
   - 内容：每个 metric 的 effective-video score 和 end-to-end score。
   - 来源：`metric_scores.png` 可作为草图来源，但论文图需要重画。

4. **Figure 4: Validity and effective rates by metric**
   - 位置：Experiments / Quantitative Evaluation。
   - 内容：physics valid rate、tracking pass rate、effective-video rate。
   - 来源：`metric_rates.png` 可作为草图来源。

## 必须补的表

1. **Table 1: Benchmark comparison**
   - 位置：Related Work。
   - 当前为 checkmark 风格矩阵。
   - 后续需要检查引用是否最终准确。

2. **Table 2: Task taxonomy**
   - 位置：Benchmark Datasets。
   - 当前已有初稿。
   - 已将 restitution coefficient 的 expected objects 改为 1，表示单物体与固定表面碰撞；two-object collision 留给 momentum conservation。
   - 待补：每个 metric 的最终数量、单/双物体数量。

3. **Table 3: Default tolerance values**
   - 位置：Evaluation Metrics。
   - 当前来自 `scripts/score_results.py`。
   - 后续确认最终 tau 设置。

4. **Table 4: Main model comparison**
   - 位置：Experiments / Quantitative Evaluation。
   - 来源：每个模型的 `score_summary.json`。
   - 列：model、end-to-end score、effective score、effective rate、tracking pass、physics valid、median relative error。

5. **Table 5: Mask-reliability thresholds**
   - 位置：Appendix / Planarity and Mask Reliability Score。
   - 来源：`scripts/score_results.py` 中的 mask score thresholds。

## 需要补全文字的位置

1. **Abstract**
   - 补最终数据规模、模型数量、最好模型、最难指标、discard rate、headline score。

2. **Introduction**
   - 最后一条贡献可以替换成最终 empirical finding。

3. **Benchmark Datasets**
   - 补最终 metadata 统计：总视频数、每个指标数量、object count 分布、scale/calibration 分布。

4. **Evaluation Metrics**
   - 最终确认 tau、weak-valid multiplier、tracking policy、mask policy 是否就是论文主结果设置。
   - 正文已经引入 \(Q_{\mathrm{mask}}\)，详细计算和阈值在附录。

5. **Experiments / Discard Rate**
   - 写出哪些指标 discard 高，主要原因是什么。

6. **Experiments / Quantitative Evaluation**
   - 写出模型排名、最强/最弱指标、effective 与 end-to-end 的差异。

7. **Conclusion**
   - 用最终实验发现替换 TODO。

## 论文图注意事项

- Python 自动生成图只能作为数据源和草图。
- 论文图需要统一字体、字号、颜色和图例风格。
- metric 名称要短且一致。
- Figure 2 必须说明 exclusion reasons 是 multi-label。
- Figure 3/4 应与当前打分脚本的逻辑一致：mask 影响 score，但不影响 effective-video count。
