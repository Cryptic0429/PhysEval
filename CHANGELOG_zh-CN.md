# PhysEval 项目调整记录

更新：2026-09-27。本文件统一记录项目整理、代码修改、验证及待办，不再单独维护项目审查、Qmask 和弹簧专题文档。路径除特别说明外均相对仓库根目录；逐样本数据保留在输出目录。

发布说明：工作区的重评分 CSV/JSON、视频、缓存、论文修订草稿和历史 rebuttal 产物属于本地研究数据，不随代码发布分支提交；本文中的输出路径用于本地复核。

## 1. 目录、入口与发布材料

| 位置 | 已做调整 |
| --- | --- |
| 根目录 `README.md`、`docs/README.md` | 整理项目导航，明确代码、数据、历史结果、验证材料和文档的职责。 |
| `benchmark/metadata/` | 集中提示词/评测工作簿和 CSV 导出，补充规模、字段、单位、任务划分及中英文说明。 |
| `docs/manuscript/`、`docs/references/`、`docs/rebuttal/` | 分别组织论文、相关工作和审稿记录；当前审稿材料集中为 `DECISION_AND_REVIEWS.md`，移除重复答复草稿及过程性材料。 |
| `scripts/` | 删除 11 个一次性 rebuttal/补充实验分析脚本；保留主跟踪、物理评估、评分入口及已有实验产物。删除清单见下。 |
| `physeval-compare/` → `compare/` | 将对比模块、脚本、测试及历史对比报告迁入主项目，可在同一 GitHub 仓库分发。 |
| `physeval.py` | 新增统一入口：`eval`、`score`、`compare`、`compare-summary`；仅显式调用对比命令时加载 compare。 |
| `scripts/report_io.py` | 抽出分析报告的 CSV/JSON 写入工具，避免弹簧和敏感性分析依赖历史 Qmask 对照脚本。 |
| `requirements-compare.txt`、`compare/` 内配置/资源脚本 | 对比依赖独立安装；对齐主项目元数据、视频及模型资源路径，保留原三种检测器/跟踪器模式。 |
| `.gitignore`、`compare/.gitignore` | 排除缓存、临时目录、本地环境、第三方源码/权重及新生成对比输出；历史对比报告保留。 |
| `README.md`、`QUICKSTART_zh-CN.md`、`SCORING_USAGE.md` | 更新结构、安装、可选对比、评分口径、协议参数和输出说明。 |

删除的一次性脚本（均原位于 `scripts/`）：

```text
analyze_morpheus_full_results.py       analyze_open_video_outputs.py
build_morpheus_semantic_contact_sheet.py
build_rebuttal_asset_manifest.py      build_rebuttal_validation_table.py
create_semantic_review_template.py    plot_rebuttal_dashboard.py
rebuttal_bootstrap_stats.py           rebuttal_prompt_variance.py
rebuttal_sensitivity.py               summarize_tracking_threshold_sensitivity.py
```

## 2. Qmask 与 compare 数据一致性

以下路径相对 ``。

| 位置 | 已做调整 |
| --- | --- |
| `physics_eval/quality/{protocol,scoring,adapters,diagnostics}.py` | 新增共享协议校验、纯评分、轨迹适配和质量诊断；主项目与 compare 共用。记录配置哈希和证据来源。 |
| `configs/scoring/legacy_v1.json` | 冻结旧规则：γ=0.35、指标/物体取最小值，保留历史兼容行为。 |
| `configs/scoring/grouped_v2_candidate.json` | 当前选定的 Qmask 规则：γ=1，九项指标分三组等权，组内及物体间平均；中心跳步仅作诊断。 |
| `scripts/score_results.py` | 新评分默认使用 `grouped_v2_candidate`；缺证据不补满分，报告按协议分目录并保存配置。 |
| `scripts/quality/check_mask_area_stability.py`、`physics_eval/run_video_batch.py` | 将原诊断算法移入共享模块，CLI/批处理复用；保存诊断版本和来源，支持磁盘及内存掩膜证据。 |
| `compare/physeval_compare/{metrics,output}.py` | 统一跟踪门控和 Qmask；输出 schema 2（每帧一条、物体嵌套、bbox 含端点），修正多物体帧数口径；兼容读取旧扁平结构及旧 bbox。 |
| `compare/physeval_compare/{runner,summary}.py` | 增加 Qmask 协议选项、运行配置及协议标识；拒绝混合协议汇总、缺证据成功项及不同协议缓存的静默复用。旧 compare 公式可显式选择 `compare_legacy_v1`。 |
| `scripts/{analyze_qmask_protocols,validate_qmask_synthetic}.py` | 保留历史协议对照产物；当前合成检查只核验 `grouped_v2_candidate`，后续参数分析固定该协议。 |
| `physics_eval/tests/test_qmask_protocols.py` | 13 项测试覆盖映射、缺项、配置、双物体帧数、证据适配、主流程/compare 一致性、门控及 grouped 默认值。 |

**历史记录：**旧协议对照曾重评分 3,000 条，有效数为 1,047。此对照仅作为历史产物，不再用于当前 Qmask 分析或协议选择；当前新评分以分组协议为准。

历史本地输出（不随代码提交）：`outputs/qmask_protocol_review/2026-09-26-complete/`。不要将其中的旧协议差值作为当前结果。

## 3. 弹簧周期：固定周期质量诊断与当前实现

以下路径相对 ``。

| 位置 | 已做调整 |
| --- | --- |
| `physics_eval/utils/fitting.py` | 保留原周期估计与 `(T, confidence, method)` 三元组返回接口；严格质量筛选分支已移除。旧估计器仍按原峰距和置信度启发式运行。 |
| `physics_eval/utils/period_quality.py` | 仅对既有周期 T 做固定周期拟合诊断；峰值与自相关来源共用诊断规则，不搜索或替换 T。质量分不解释为概率。 |
| 严格周期筛选试验 | 已从当前代码入口、专属验证脚本和专属测试中移除；旧输出仅作为历史记录保留。 |
| 固定 T 诊断 | 锁定原 T，仅拟合质量诊断，避免用另一个周期的良好拟合为旧 k 提权。 |
| `physics_eval/evaluators/spring.py` | 新增 v3：保留旧有效资格、T/k、来源、单位和基础误差分，只调整 1.0/0.8 权重；诊断状态独立保存，诊断不足/异常不新增剔除。保留 k=4π²m/T²，并检查质量计算输入及质量字段。 |
| `physics_eval/{run_batch_eval,run_video_batch}.py` | 增加 `--spring-period-protocol`，只作用于弹簧；v3 禁止与会剔除弱有效样本的 `--strict` 联用。 |
| `scripts/analyze_spring_period.py` | 不再提供严格周期筛选或协议选择参数；独立目录重算只用固定 T 的可选 v3 诊断，并逐条断言有效成员、测量、基础分和 grouped Qmask 不变。 |
| `scripts/{spring_synthetic_cases,validate_spring_confidence}.py` | 保留固定 T 质量诊断的合成用例，覆盖周期、时长、FPS、噪声、阻尼、缺帧及非周期运动。 |
| `physics_eval/tests/test_spring_confidence.py` | 7 项测试覆盖 v3 状态权重、输入异常与合成诊断；严格周期筛选测试已移除。 |

本轮相关测试：Qmask 13 项 + 弹簧置信度 7 项，共 **20 项通过**。Qmask 默认值测试直接核验配置加载与评分 CLI 的默认协议。

**历史重算结果须与当前默认区分：**

| 方案 | 弹簧有效数 | 原 185 条峰值记录 | 定位 |
| --- | --- | --- | --- |
| 旧严格周期筛选试验 | 138→30 | 12 valid / 46 weak_valid / 127 invalid | 已从当前实现移除；数字只记录历史试验 |
| 历史 `period_confidence_v3_candidate` 重算 | **138→138，成员相同** | 12 valid / 173 weak_valid，无新增剔除 | v3 仅作可选诊断；弹簧默认估计仍走 legacy 路径 |

上述历史重算中全项目有效数保持 1,047，其他 2,700 条结果不变；固定 138 条弹簧有效样本均分为 4.611093→4.614528。300 条弹簧记录中 280 条有轨迹，20 条缺轨迹的旧失败项保持失败。切换 Qmask 默认值不会自动重算或改写这些存档结果。

当前本地输出（不随代码提交）：`outputs/spring_period_review/2026-09-26/confidence_only_v3/`，严格旧试验位于相邻 `history_release/`。

以下为两份历史**独立对照**的有效均分：Qmask 列为全指标宏平均，弹簧列仅为弹簧指标；两者不能相加，也不能解释为当前统一默认配置的重算结果。

| 模型 | Qmask 旧→候选 | 弹簧旧→v3 | 弹簧有效数 |
| --- | --- | --- | ---: |
| CogVideoX-5b | 0.1423→11.9586 | 6.5352e-7→5.2281e-7 | 25 |
| HunyuanVideo | 9.5916→12.9637 | 3.6908→3.9524 | 15 |
| jimeng | 11.4613→15.4697 | 6.8690→6.7319 | 32 |
| kling | 9.0544→15.4869 | 4.2463→4.2412 | 29 |
| wan2.1 | 4.4227→12.4395 | 5.3271→5.3271 | 12 |
| wan2.2 | 11.1767→14.5301 | 6.9636→7.0070 | 25 |

弹簧 v3 历史重算对原有效候选共升级 12 条、降权 62 条，不剔除；其中通过跟踪门控的 138 条内部为 6 条升级、46 条降权、86 条不变。按分差>1e-12计，300 条中 4 条升分、12 条降分、284 条无明显变化。此 v3 目前仍需通过 `--spring-period-protocol period_confidence_v3_candidate` 显式启用。

valid 质量条件集中于 `PeriodRules`：观测≥3周期、振幅支持≥2.5周期、拟合解释度≥0.85、间隔不一致度≤0.15、覆盖率≥0.85、最大缺口≤0.25周期、每周期≥12点、至少两个可比峰间隔。v3 中条件不满足仅使用弱权重。诊断锁定旧 T，因此旧估计器的长度相关峰距、周期候选误判及倍周期误估仍可能存在；这些问题不属于当前已修复范围。

## 4. 文档整理与当前边界

- 项目审查、Qmask 与弹簧专题文档已精简并入本文件，原三个文件删除；文档目录和使用指南统一引用本记录。
- **当前默认 Qmask 为 `grouped_v2_candidate`**；Qmask `legacy_v1` 及 compare 旧公式仅保留显式兼容入口，不用于当前分析。弹簧周期置信度 v3 可选且须显式启用；默认弹簧估计器仍使用旧周期估计。尚未替换论文主表。联合评分基准及参数敏感性见第 7 节。
- 重算写入新目录，保留原始 `result.json` 和历史报告，保存输入指纹、配置及验证结果；未重跑 GPU 跟踪。
- 尚未完成：Qmask 独立人工/真值校准和实际测量窗口扩展；弹簧真实视频真值校准及旧 T 的误估修复；项目审查中的动量质量绑定等其他估计器问题。
- 工作区既有删除、原始视频缺失及临时文件不自动计为本轮算法修改；本记录不将未验证的问题标为已修复。

后续更新沿用“位置 → 修改 → 默认/可选 → 验证及影响”的格式追加，已否定或替代的方案保留状态说明。

## 5. 尚未解决的审查事项

以下为待办或适用边界，不是已完成修改：

| 范围 | 待处理事项 |
| --- | --- |
| 动量 | 轨迹按记录数排序，未稳定绑定物体身份与质量；不等质量样本存在错配风险。 |
| 输入与计数 | 默认元数据路径及 CSV/XLSX 读取需统一复核；跟踪对象数不能代表画面全部对象，也不能保证识别中途增殖。 |
| 其他估计器 | 密度物理范围、能量事件/初速度、摩擦减速、黏度终端状态及模型适用性检查仍不足。 |
| 计量与汇总 | 需验证单位/尺度变化下的状态稳定性；明确零目标动量的相对误差特例、指标宏平均与视频合并平均、无有效样本指标的处理。 |
| 真值验证 | 补分层真值、真实视频周期、相机/尺度/时间基准、密度/黏度可辨识性、区间估计和生成种子敏感性。 |
| 工程发布 | 完善依赖/上游版本固定、完整缓存指纹、运行清单、其他估计器测试、人工工作簿工具可复现性、许可/引用/打包/CI；文稿占位内容另行处理。 |

## 6. 复现与数据入口

从仓库根目录运行；重分析必须指定新输出目录：

```powershell
python scripts/validate_qmask_synthetic.py --output outputs/grouped_qmask_synthetic.csv
python scripts/analyze_spring_period.py --output-dir outputs/spring_new
python scripts/validate_spring_confidence.py --output-dir outputs/spring_validation_new
python scripts/analyze_scoring_sensitivity.py --output-dir outputs/scoring_sensitivity_new
```

评分默认采用 `grouped_v2_candidate`，通常无需指定协议参数。需要试用弹簧置信度诊断时，显式传入 `--spring-period-protocol period_confidence_v3_candidate`；弹簧默认估计方式仍不变。

历史分析目录保留模型汇总、逐视频表、配置、输入指纹及验证结果；旧 Qmask 协议矩阵是归档结果，不作为当前分析入口。

## 7. 当前方案的参数敏感性（2026-09-27）

Qmask 固定使用 `grouped_v2_candidate`；参数敏感性实验只变动其 γ 等参数，不再与 Qmask `legacy_v1` 做协议择优对照。该敏感性分析显式选用弹簧 `period_confidence_v3_candidate` 作为实验诊断覆盖层，**不代表它是弹簧默认估计器**。基准 γ=1、weak_valid 权重=0.8、R² weak/valid 阈值=0.65/0.85。仅使用现存结果和 300 条已保存弹簧 v3 候选，未重跑视频生成、GPU 跟踪或物理参数估计；论文主表未更改。

新增可复现脚本：`scripts/analyze_scoring_sensitivity.py`。共 25 个设置（含基准与三组基准重复），六模型各 500 条，共 75,000 条评分；每次仅改变一个因素。

| 分析 | 设置 | 有效数与分数影响 | 排名影响 |
| --- | --- | --- | --- |
| Qmask γ | 0.35、0.5、0.75、1、1.25、1.5、2；聚合、组件和门控固定 | 有效集合保持 1,047；γ 越大，中间质量分量的惩罚越弱。相对基准，各模型 E2E 变化范围为 −10.37% 至 +5.81%。 | E2E 排名完全不变，ρ=1；有效宏平均前两名 Jimeng/Kling 交换，ρ=0.943–1。 |
| R² 状态阈值 | weak=0.60/0.65/0.70 × valid=0.80/0.85/0.90；仅速度、加速度、重力、密度，共 1,200 条 | 四指标有效数 439→454/439/429（+3.42%/0/−2.28%）；全项目 1,047→1,062/1,047/1,037（+1.43%/0/−0.96%）。valid 截断只改变 valid/weak 权重分配。 | 九种设置下，全指标及四指标 E2E 排名均不变，ρ=1；有效宏平均仍有 Jimeng/Kling 交换，ρ=0.943–1。 |
| weak_valid 权重 | 0、0.2、0.4、0.6、0.7、0.8、0.9、1；状态和有效集合固定 | 1,047 条有效样本中 314 条 weak_valid。0.6–1.0 时各模型 E2E 相对基准最大变化约 ±3.34% 至 ±13.19%。权重 0 仅取消弱有效的分数贡献，不剔除样本。 | E2E 排名在整个 [0,1] 内无交叉；有效宏平均有五个交叉点，其中 Jimeng/Kling 在 w=0.782833 交叉。 |

联合基准的 E2E 排名为 `jimeng > wan2.2 > HunyuanVideo > kling > CogVideoX-5b > wan2.1`；有效宏平均为 `kling > jimeng > wan2.2 > HunyuanVideo > wan2.1 > CogVideoX-5b`。w=0.8 时 Kling/Jimeng 有效宏平均为 15.4791/15.4465，仅差 0.0326 分，不能称有效视频排名第一名对参数选择稳健。这里的端到端分数与 rebuttal 的旧评分汇总不是同一个配置，不能直接混用。

单个模型指标的 E2E 分差在 R² 网格中最大约 1.52 分（Wan2.2 密度）；宏平均稳定会掩盖局部变化。另报告九种阈值共同有效的 1,037 条视频，辅助区分权重与样本构成变化。敏感性结果不构成参数真值最优性校准。

最终本地输出（不随代码提交）：`outputs/scoring_sensitivity/2026-09-27-final/`，含中文报告、模型/指标/逐视频 CSV、状态与排名汇总、共同有效集合、权重线性系数和精确交叉点，以及 E2E/有效宏平均两套 PNG/PDF/SVG 图。`2026-09-27-current/` 是首轮产物；最终引用使用 `-final/`。

验证：四指标基准状态回放 1,200/1,200 一致；γ/权重实验有效成员不变；弱权重评分线性、γ 单调性、R² 未涉及的指标逐条不变、基准重复一致及输入/代码哈希检查均通过。具体证据见 `verification.json` 与 `input_manifest.json`。
