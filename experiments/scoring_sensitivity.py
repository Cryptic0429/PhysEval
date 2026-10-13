"""One-factor sensitivity on saved evidence, with grouped Qmask fixed.

Never modifies input results or changes the production default. Every scenario
uses the production scorer; physical R2 cutoffs apply only to its four consumers.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
from physics_eval.quality.protocol import load_protocol, protocol_hash
from physics_eval import scoring as scorer
from physics_eval.cli import parse_score_args
from physics_eval.io.reports import write_csv, write_json

AFFECTED = {"velocity", "acceleration", "gravity", "density_from_initial_acceleration"}
GAMMAS = (0.35, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
WEIGHTS = (0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 1.0)
WEAK_CUTOFFS = (0.60, 0.65, 0.70)
VALID_CUTOFFS = (0.80, 0.85, 0.90)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def scenarios():
    base = dict(gamma=1.0, weak_weight=0.8, weak_cutoff=0.65, valid_cutoff=0.85)
    runs = [dict(scenario="baseline", family="baseline", **base)]
    for gamma in GAMMAS:
        runs.append(dict(scenario=f"gamma_{gamma:g}", family="gamma", **{**base, "gamma": gamma}))
    for weight in WEIGHTS:
        runs.append(dict(scenario=f"weight_{weight:g}", family="weak_weight", **{**base, "weak_weight": weight}))
    for weak in WEAK_CUTOFFS:
        for valid in VALID_CUTOFFS:
            runs.append(dict(scenario=f"r2_{weak:g}_{valid:g}", family="r2", **{**base, "weak_cutoff": weak, "valid_cutoff": valid}))
    return runs


def status_at(data, weak, valid):
    physics = data["physics_result"]
    if data["metric"] not in AFFECTED or physics["Status"] == "failed":
        return physics["Status"]
    r2 = scorer.as_float(physics.get("Fit_R2"))
    if r2 is None:
        raise ValueError("Non-failed R2 consumer missing a finite Fit_R2")
    # Preserve the evaluator's non-R2 zero-gravity exception, if it exists.
    reference = scorer.as_float(data.get("physics_extra", {}).get("reference_g_mps2"))
    measured = scorer.as_float(physics.get("Measured_Value"))
    if data["metric"] == "gravity" and reference == 0 and measured is not None and measured < 0.2:
        return "valid"
    return "valid" if r2 >= valid else "weak_valid" if r2 >= weak else "invalid"


def order(items, field):
    return [r["model"] for r in sorted(items, key=lambda r: (-r[field], r["model"]))]


def rank_rho(a, b):
    # There are no tied model scores in the analyzed cohort; checked below.
    n = len(a)
    return 1 - 6 * sum((a.index(m) - b.index(m)) ** 2 for m in a) / (n * (n*n - 1))


def reversed_pairs(a, b):
    return sum(b.index(a[i]) > b.index(a[j]) for i in range(len(a)) for j in range(i+1, len(a)))


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT.parent / "results")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Use a fresh output directory; existing analyses are immutable")
    out.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol("grouped_v2_candidate")
    design = dict(qmask=protocol, qmask_hash=protocol_hash(protocol), spring_confidence_threshold=0.20,
                  r2_metrics=sorted(AFFECTED), scenarios=scenarios(),
                  aggregation="metric macro average via production scorer; empty effective cells excluded from effective macro",
                  design="one factor at a time; all other parameters fixed; no GPU retracking",
                  weak_zero_policy="zero contribution, eligibility preserved; not a strict exclusion experiment")
    write_json(out / "design.json", design)
    defaults = parse_score_args(["--no-plots", "--scoring-protocol", "grouped_v2_candidate"])
    paths_by_model, data, manifest = {}, {}, []
    spring_n = 0
    for model_dir in sorted(args.results_root.iterdir()):
        paths = scorer.discover_result_files(model_dir, None) if model_dir.is_dir() else []
        if not paths:
            continue
        complete = scorer.check_completeness(paths, defaults.metadata)
        if any(complete[k] for k in ("missing_ids", "unexpected_ids", "duplicate_ids")):
            raise ValueError(f"Incomplete model {model_dir.name}: {complete}")
        paths_by_model[model_dir.name] = paths
        for path in paths:
            value = json.loads(path.read_text(encoding="utf-8"))
            manifest.append(dict(path=str(path.resolve()), sha256=sha(path), role="original_result"))
            if value["metric"] == "spring_constant":
                physics = value["physics_result"]
                if physics["Status"] in scorer.PHYSICS_VALID_STATUSES:
                    confidence = scorer.as_float(physics.get("Fit_R2"))
                    if confidence is None:
                        raise ValueError(f"Spring result missing period confidence: {path}")
                    expected = "valid" if confidence >= 0.20 else "weak_valid"
                    if physics["Status"] != expected:
                        raise ValueError(f"Spring status inconsistent with period confidence: {path}")
                spring_n += 1
            data[path] = value
    assert len(data) == 3000 and spring_n == 300 and len(paths_by_model) == 6
    affected_paths = [p for p, d in data.items() if d["metric"] in AFFECTED]
    assert len(affected_paths) == 1200
    assert all(status_at(data[p], .65, .85) == data[p]["physics_result"]["Status"] for p in affected_paths)
    for source in (Path(__file__), ROOT / "physics_eval/scoring.py", ROOT / "physics_eval/quality/mask_score.py",
                   ROOT / "physics_eval/quality/estimation_checks.py", ROOT / "configs/scoring/grouped_v2_candidate.json",
                   Path(defaults.metadata)):
        manifest.append(dict(path=str(source.resolve()), sha256=sha(source), role="code_config_or_metadata"))
    write_json(out / "input_manifest.json", manifest)
    print(f"Loaded {len(data)} videos, including {spring_n} spring results; baseline R2 replay 1200/1200", flush=True)
    current = dict(data)
    summaries, metrics, per_video, global_rows = [], [], [], []
    saved_runs = {}
    baseline = None
    for scenario in design["scenarios"]:
        run = copy.copy(defaults)
        run.weak_valid_multiplier = scenario["weak_weight"]
        run.qmask_protocol = copy.deepcopy(protocol)
        run.qmask_protocol["gamma"] = scenario["gamma"]
        current = dict(data)
        for path in affected_paths:
            value = data[path]
            status = status_at(value, scenario["weak_cutoff"], scenario["valid_cutoff"])
            if status != value["physics_result"]["Status"]:
                physics = {**value["physics_result"], "Status": status}
                physics["Failure_Reason"] = "fit_failed" if status == "invalid" else "low_fit_r2" if status == "weak_valid" else ""
                current[path] = {**value, "physics_result": physics}
        rows, path_rows = [], {}
        for model, paths in paths_by_model.items():
            run.model_name = model
            run.result_root = str(args.results_root / model)
            model_rows = [scorer.score_data(current[p], p, scorer.DEFAULT_TOLERANCES, run) for p in paths]
            unavailable = [r for r in model_rows if not r["score_available"]]
            if unavailable:
                write_json(out / "unavailable.json", unavailable)
                raise ValueError("Required evidence unavailable: refusing formal ranking")
            path_rows.update(zip(paths, model_rows))
            rows.extend(model_rows)
            summary = scorer.summarize(model_rows)
            affected = scorer.summarize([r for r in model_rows if r["metric"] in AFFECTED])
            summaries.append(dict(**scenario, model=model, n_total=len(model_rows), n_effective=summary["n_effective_for_score"],
                n_weak_effective=sum(r["effective_for_score"] and r["status"] == "weak_valid" for r in model_rows),
                effective_macro=summary["overall_effective_video_score"], end_to_end_macro=summary["overall_end_to_end_score"],
                affected_n_effective=affected["n_effective_for_score"], affected_end_to_end_macro=affected["overall_end_to_end_score"],
                mean_qmask_effective=statistics.mean(r["mask_quality_score"] for r in model_rows if r["effective_for_score"])))
            metrics.extend(dict(**scenario, model=model, **m) for m in summary["metrics"])
        if baseline is None:
            baseline = path_rows
        for path, row in path_rows.items():
            before = baseline[path]
            assert row["measured_value"] == before["measured_value"]
            assert row["tracking_quality_status"] == before["tracking_quality_status"]
            if scenario["family"] in {"gamma", "weak_weight"}:
                assert row["effective_for_score"] == before["effective_for_score"]
                assert row["status"] == before["status"] and row["base_score"] == before["base_score"]
            if scenario["family"] == "weak_weight":
                assert row["mask_quality_score"] == before["mask_quality_score"]
                expected = (before["base_score"] * before["mask_quality_score"] *
                            (scenario["weak_weight"] if before["status"] == "weak_valid" else 1)) if before["effective_for_score"] else 0
                assert abs(row["adjusted_score"] - expected) < 1e-10
            if scenario["family"] == "gamma" and before["effective_for_score"]:
                delta = row["adjusted_score"] - before["adjusted_score"]
                assert delta >= -1e-10 if scenario["gamma"] >= 1 else delta <= 1e-10
            if scenario["family"] == "r2" and row["metric"] not in AFFECTED:
                assert row["adjusted_score"] == before["adjusted_score"]
            if (scenario["gamma"], scenario["weak_weight"], scenario["weak_cutoff"], scenario["valid_cutoff"]) == (1,.8,.65,.85):
                assert row["adjusted_score"] == before["adjusted_score"] and row["status"] == before["status"]
            per_video.append(dict(**scenario, model=row["model"], metric=row["metric"], prompt_id=row["prompt_id"],
                original_status=data[path]["physics_result"]["Status"], status=row["status"],
                fit_r2=current[path]["physics_result"].get("Fit_R2"), tracking_pass=row["tracking_quality_pass"],
                effective=row["effective_for_score"], baseline_effective=before["effective_for_score"],
                measured_value=row["measured_value"], base_score=row["base_score"], status_multiplier=row["status_multiplier"],
                qmask=row["mask_quality_score"], adjusted_score=row["adjusted_score"], delta_from_baseline=row["adjusted_score"]-before["adjusted_score"]))
        saved_runs[scenario["scenario"]] = path_rows
        counts = Counter(current[p]["physics_result"]["Status"] for p in affected_paths)
        global_rows.append(dict(**scenario, n_effective=sum(r["effective_for_score"] for r in rows),
            affected_n_effective=sum(path_rows[p]["effective_for_score"] for p in affected_paths),
            affected_n_physics_pass=counts["valid"]+counts["weak_valid"], affected_valid=counts["valid"],
            affected_weak_valid=counts["weak_valid"], affected_invalid=counts["invalid"], affected_failed=counts["failed"],
            status_changed=sum(path_rows[p]["status"] != baseline[p]["status"] for p in path_rows),
            newly_effective=sum(r["effective_for_score"] and not baseline[p]["effective_for_score"] for p,r in path_rows.items()),
            no_longer_effective=sum(not r["effective_for_score"] and baseline[p]["effective_for_score"] for p,r in path_rows.items())))
        print(f"Scored {scenario['scenario']}: effective {global_rows[-1]['n_effective']}", flush=True)
    base_summary = [r for r in summaries if r["family"] == "baseline"]
    fields = {"e2e": "end_to_end_macro", "effective": "effective_macro", "affected_e2e": "affected_end_to_end_macro"}
    base_orders = {name: order(base_summary, field) for name, field in fields.items()}
    for global_row in global_rows:
        items = [r for r in summaries if r["scenario"] == global_row["scenario"]]
        for name, field in fields.items():
            assert len({r[field] for r in items}) == len(items), "Tied scores require tie-aware Spearman"
            ranked = order(items, field)
            global_row[name + "_order"] = " > ".join(ranked)
            global_row[name + "_rho"] = rank_rho(base_orders[name], ranked)
            global_row[name + "_reversed_pairs"] = reversed_pairs(base_orders[name], ranked)
            for item in items:
                item[name + "_rank"] = ranked.index(item["model"]) + 1
        for item in items:
            before = next(r for r in base_summary if r["model"] == item["model"])
            item["e2e_delta"] = item["end_to_end_macro"] - before["end_to_end_macro"]
            item["effective_delta"] = item["effective_macro"] - before["effective_macro"]
    # Intersection across all threshold settings: same-video diagnostic separated
    # from each scenario's own survivor-conditioned effective macro.
    r2_names = [s["scenario"] for s in design["scenarios"] if s["family"] == "r2"]
    common = {p for p in data if all(saved_runs[n][p]["effective_for_score"] for n in r2_names)}
    common_summary = []
    for name in r2_names:
        for model, paths in paths_by_model.items():
            common_rows = [saved_runs[name][p] for p in paths if p in common]
            summary = scorer.summarize(common_rows)
            common_summary.append(dict(scenario=name, model=model, n_common=len(common_rows),
                common_effective_macro=summary["overall_effective_video_score"]))
    # All weight curves are affine because eligibility and Qmask stay fixed.
    crossings = []
    coefficients = []
    for model in paths_by_model:
        lo = next(r for r in summaries if r["scenario"] == "weight_0" and r["model"] == model)
        hi = next(r for r in summaries if r["scenario"] == "weight_1" and r["model"] == model)
        coefficients.append(dict(model=model, e2e_intercept=lo["end_to_end_macro"],
            e2e_slope=hi["end_to_end_macro"]-lo["end_to_end_macro"],
            effective_intercept=lo["effective_macro"], effective_slope=hi["effective_macro"]-lo["effective_macro"]))
    for i, a in enumerate(coefficients):
        for b in coefficients[i+1:]:
            for kind in ("e2e", "effective"):
                delta_slope = a[kind+"_slope"]-b[kind+"_slope"]
                if abs(delta_slope) > 1e-12:
                    weight = (b[kind+"_intercept"]-a[kind+"_intercept"])/delta_slope
                    if 0 <= weight <= 1:
                        crossings.append(dict(score_type=kind, model_a=a["model"], model_b=b["model"], crossing_weight=weight))
    write_csv(out / "model_summary.csv", summaries)
    write_csv(out / "metric_summary.csv", metrics)
    write_csv(out / "per_video_sensitivity.csv", per_video)
    write_csv(out / "scenario_summary.csv", global_rows)
    write_csv(out / "threshold_common_cohort.csv", common_summary)
    write_csv(out / "weight_coefficients.csv", coefficients)
    write_json(out / "weight_rank_crossings.json", crossings)
    write_json(out / "summary.json", dict(baseline=base_summary, scenarios=global_rows, weight_crossings=crossings))
    unchanged = all(sha(Path(m["path"])) == m["sha256"] for m in manifest)
    assert unchanged
    write_json(out / "verification.json", dict(n_inputs=len(data), n_spring_results=spring_n,
        n_scenarios=len(design["scenarios"]), n_scores=len(per_video), r2_baseline_status_replay=1200,
        baseline_effective=sum(r["effective_for_score"] for r in baseline.values()),
        no_unavailable_scores=True, all_input_files_unchanged=unchanged,
        gamma_and_weight_effective_membership_preserved=True, weak_weight_affine_score_check=True,
        gamma_score_monotonicity_check=True, threshold_unaffected_metrics_exactly_preserved=True,
        baseline_repeated_scenarios_identical=True, independent_quality_calibration="not performed"))
    make_report(out, summaries, metrics, global_rows, coefficients, crossings, common_summary)
    make_plot(out, summaries, global_rows)
    print(f"Reports: {out}", flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(parse_args(argv))


def make_report(out, summaries, metrics, global_rows, coefficients, crossings, common_summary):
    baseline = [r for r in summaries if r["family"] == "baseline"]
    base_global = next(r for r in global_rows if r["family"] == "baseline")
    model_order = [r["model"] for r in baseline]
    lines = ["# 评分参数敏感性", "",
        "基准：九项掩膜指标按三组等权聚合，γ=1；weak_valid权重=0.8，四项共享R²阈值为0.65/0.85。弹簧使用峰值或自相关周期估计，周期置信度达到0.20为valid，否则可测样本为weak_valid。",
        "直接读取六模型各500条保存结果，保持物理测量与跟踪固定，每次仅改变一个评分因素。弹簧的周期置信度不套用共享R²阈值。",
        "分数单位为0–100。E2E将未通过门控的样本记0，按十指标平均；条件宏平均只使用各指标的有效样本，排除无有效样本的指标。", "",
        "| 模型 | 有效数 | 弱有效数 | 条件宏平均 | E2E宏平均 |",
        "| --- | ---: | ---: | ---: | ---: |"]
    for r in baseline:
        lines.append(f"| {r['model']} | {r['n_effective']} | {r['n_weak_effective']} | {r['effective_macro']:.4f} | {r['end_to_end_macro']:.4f} |")
    lines.extend(["", f"基准有效数：{base_global['n_effective']:,}。排名与分数变化均由本次输入计算。", ""])
    for family, title in (("gamma", "γ：固定聚合公式和门控"),
                           ("r2", "R²：速度、加速度、重力和密度"),
                           ("weak_weight", "弱有效权重：固定状态与有效集合")):
        rows = [r for r in global_rows if r["family"] == family]
        lines.extend([f"## {title}", "",
            "| 设置 | 全指标有效数 | E2E排名ρ | 条件排名ρ | E2E排名 | 条件排名 |",
            "| --- | ---: | ---: | ---: | --- | --- |"])
        for g in rows:
            label = (f"γ={g['gamma']:g}" if family == "gamma" else
                     f"w={g['weak_weight']:g}" if family == "weak_weight" else
                     f"{g['weak_cutoff']:.2f}/{g['valid_cutoff']:.2f}")
            lines.append(f"| {label} | {g['n_effective']} | {g['e2e_rho']:.3f} | {g['effective_rho']:.3f} | {g['e2e_order']} | {g['effective_order']} |")
        lines.extend(["", "| 设置 | " + " | ".join(model_order) + " |",
            "| --- | " + " | ".join(["---:"] * len(model_order)) + " |"])
        for g in rows:
            vals = [next(r for r in summaries if r["scenario"] == g["scenario"] and r["model"] == m)["effective_macro"] for m in model_order]
            lines.append(f"| {g['scenario']} | " + " | ".join(f"{v:.4f}" for v in vals) + " |")
        lines.extend(["", "上表为条件宏平均；E2E宏平均与相对基准的差值见model_summary.csv。", ""])
    lines.extend(["## 权重交叉点", "",
        "固定有效集合时，分数随弱有效权重满足S(w)=A+B·w。w=0保留资格，仅将弱有效贡献置零。以下列出[0,1]内由本次输入算出的交叉点。", "",
        "| 分数口径 | 模型对 | 交叉权重 |", "| --- | --- | ---: |"])
    for c in crossings:
        lines.append(f"| {c['score_type']} | {c['model_a']} / {c['model_b']} | {c['crossing_weight']:.6f} |")
    if not crossings:
        lines.append("| — | 无交叉点 | — |")
    lines.extend(["", "## 解释与核验", "",
        "γ与弱有效权重只改变评分惩罚，分数上升不表示视频、跟踪或物理测量改善。R²弱阈值改变有效集合，valid阈值主要改变valid/weak分配。threshold_common_cohort.csv单独报告共同有效样本，便于识别样本构成差异。",
        "经验排名稳定不表示绝对得分不敏感；这些扫描不校准参数的真值最优性，也不测量固定提示的生成种子方差。",
        "design.json保存参数；input_manifest.json保存输入和代码哈希；model_summary.csv、metric_summary.csv、per_video_sensitivity.csv与scenario_summary.csv保存完整结果；verification.json保存不变量检查。",
        "所有输入只读，不覆盖原始结果。"])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def make_plot(out, summaries, global_rows):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ("#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9")
    models = sorted({r["model"] for r in summaries})
    for kind, score_field, ylabel in (("e2e", "end_to_end_macro", "E2E macro score (0-100)"),
                                     ("effective", "effective_macro", "Effective macro score (0-100)")):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
        for model, color in zip(models, colors):
            for ax, family, xfield in ((axes[0], "gamma", "gamma"), (axes[1], "weak_weight", "weak_weight")):
                rows = sorted([r for r in summaries if r["model"] == model and r["family"] == family], key=lambda r:r[xfield])
                ax.plot([r[xfield] for r in rows], [r[score_field] for r in rows], "o-", color=color, label=model, linewidth=1.6, markersize=4)
            rows = sorted([r for r in summaries if r["model"] == model and r["family"] == "r2" and r["valid_cutoff"] == .85], key=lambda r:r["weak_cutoff"])
            axes[2].plot([r["weak_cutoff"] for r in rows], [r[score_field] for r in rows], "o-", color=color, linewidth=1.6, markersize=4)
        for ax, title, xlabel, baseline_value in zip(axes,
            ("Qmask gamma", "Weak-valid weight", "R2 weak cutoff (valid = 0.85)"),
            ("Gamma", "Weight", "Weak cutoff"), (1.0,.8,.65)):
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.axvline(baseline_value, color="#555555", linestyle="--", linewidth=.9)
            ax.grid(alpha=.2)
            ax.spines[["right", "top"]].set_visible(False)
        axes[0].set_xticks(GAMMAS)
        axes[2].set_xticks(WEAK_CUTOFFS)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower center", ncol=6, frameon=False, fontsize=9)
        fig.suptitle("One factor at a time: grouped Qmask; 3,000 videos", fontsize=13)
        fig.tight_layout(rect=(0,.09,1,.94))
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(out / f"sensitivity_{kind}.{suffix}", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
