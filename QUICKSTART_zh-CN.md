# PhysEval 使用指南

主评测与可选的检测器/跟踪器对比已整合在同一个项目中。以下命令均从
`PhysEval-main/` 目录运行。

## 目录与入口

```text
PhysEval-main/
├── physeval.py                 统一命令入口
├── benchmark/metadata/        提示词与物理评测元数据
├── physics_eval/              物理量估计与评测实现
├── scripts/                   主流程、评分与跟踪工具
├── requirements.txt           主流程依赖
├── requirements-compare.txt   可选对比依赖（包含主流程依赖）
├── compare/                   可选检测器/跟踪器对比
│   ├── physeval_compare/       对比实现
│   ├── scripts/               资源准备、汇总等工具
│   ├── tests/                 原有测试
│   ├── results/               已保存的对比报告
│   └── outputs/               新生成的对比结果，Git 忽略
├── repo/                      共享模型源码与权重，Git 忽略
└── data/t2v_videos/            本地视频输入，Git 忽略
```

| 命令 | 用途 | 是否加载 compare |
| --- | --- | --- |
| `python physeval.py eval` | 跟踪并估计物理量 | 否 |
| `python physeval.py score` | 对主评测结果评分 | 否 |
| `python physeval.py compare` | 对比检测器/跟踪器组合 | 是 |
| `python physeval.py compare-summary` | 汇总对比结果 | 是 |

每个命令后追加 `--help` 可查看该命令的参数。原来的脚本入口仍然可用。

## 主评测

先准备与机器 CUDA 环境匹配的 PyTorch，再安装主流程依赖与模型资源。
资源脚本需要 Bash 环境，例如 Linux、WSL 或 Git Bash。

```bash
python -m pip install -r requirements.txt
bash scripts/setup_sam2_assets.sh --install-sam2
python physeval.py eval --metadata benchmark/metadata/phys_t2v_bench_metadata.xlsx --video-root data/t2v_videos/model_name --output-dir batch_eval_results/model_name
python physeval.py score --result-root batch_eval_results/model_name --model-name model_name --weak-valid-multiplier 0.8
```

视频按元数据中的 `Video_File` 命名。完整参数和评分含义见
[批量评测说明](BATCH_USAGE.md)与[评分说明](SCORING_USAGE.md)。

## 按需启用 compare

只有需要比较 Faster R-CNN + SAM2、YOLO + SAM2、YOLO + TAM 时才安装额外依赖。
普通 `eval` / `score` 不导入 compare，也不需要 TAM/XMem 的依赖。

```bash
python -m pip install -r requirements-compare.txt
bash compare/scripts/setup_compare_assets.sh --install-editable
python compare/scripts/check_environment.py
python physeval.py compare --video-root data/t2v_videos/model_name --model-name model_name --dry-run
```

`--dry-run` 仅检查视频与元数据的匹配；去掉它才实际运行对比：

```bash
python physeval.py compare --video-root data/t2v_videos/model_name --model-name model_name
python physeval.py compare-summary --model-name model_name
```

compare 默认共用主项目的元数据、YOLO 权重与 `repo/` 模型资源，结果写入
`compare/outputs/model_name/`，临时帧写入 `compare/cache/`。
也可以用 `--modes yolo_sam2 yolo_tam` 选择对比组合。
对比报告描述跟踪和掩膜质量；物理量评测与评分由主流程执行。
更多参数见 [compare 文档](compare/README.md)。

## GitHub 目录说明

代码、元数据与可选模块可一起放在同一个仓库。`compare/results/` 保留了已有对比报告。
本地视频、下载的模型、缓存、环境和新生成的 compare 输出通过 `.gitignore` 排除。
原 `physeval-compare/` 已迁入 `compare/`；历史报告保留原始记录，可能包含旧运行路径。
