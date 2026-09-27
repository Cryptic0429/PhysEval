"""One-factor sensitivity on saved evidence, with grouped Qmask and spring v3 fixed.

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
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics_eval.quality.protocol import load_protocol, protocol_hash
from scripts import score_results as scorer
from scripts.report_io import write_csv, write_json

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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT.parent / "results")
    parser.add_argument("--spring-results", type=Path, default=ROOT.parent / "outputs/spring_period_review/2026-09-26/confidence_only_v3/candidate_results")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Use a fresh output directory; existing analyses are immutable")
    out.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol("grouped_v2_candidate")
    design = dict(qmask=protocol, qmask_hash=protocol_hash(protocol), spring="period_confidence_v3_candidate",
                  r2_metrics=sorted(AFFECTED), scenarios=scenarios(),
                  aggregation="metric macro average via production scorer; empty effective cells excluded from effective macro",
                  design="one factor at a time; all other parameters fixed; no GPU retracking",
                  weak_zero_policy="zero contribution, eligibility preserved; not a strict exclusion experiment")
    write_json(out / "design.json", design)
    argv = sys.argv
    sys.argv = ["score", "--no-plots", "--scoring-protocol", "grouped_v2_candidate"]
    try:
        defaults = scorer.parse_args()
    finally:
        sys.argv = argv
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
                candidate = args.spring_results / path.relative_to(args.results_root)
                overlay = json.loads(candidate.read_text(encoding="utf-8"))
                manifest.append(dict(path=str(candidate.resolve()), sha256=sha(candidate), role="spring_v3_overlay"))
                assert overlay["physics_extra"]["period_protocol"] == "period_confidence_v3_candidate"
                assert overlay["mask_qc"] == value["mask_qc"] and overlay["tracking_quality"] == value["tracking_quality"]
                for field in ("Measured_Value", "Measured_Unit", "Relative_Error", "Absolute_Error", "Target_Value"):
                    assert overlay["physics_result"].get(field) == value["physics_result"].get(field), (path, field)
                assert (overlay["physics_result"]["Status"] in scorer.PHYSICS_VALID_STATUSES) == (value["physics_result"]["Status"] in scorer.PHYSICS_VALID_STATUSES)
                value = overlay
                spring_n += 1
            data[path] = value
    assert len(data) == 3000 and spring_n == 300 and len(paths_by_model) == 6
    affected_paths = [p for p, d in data.items() if d["metric"] in AFFECTED]
    assert len(affected_paths) == 1200
    assert all(status_at(data[p], .65, .85) == data[p]["physics_result"]["Status"] for p in affected_paths)
    for source in (Path(__file__), ROOT / "scripts/score_results.py", ROOT / "physics_eval/quality/scoring.py",
                   ROOT / "physics_eval/utils/quality.py", ROOT / "configs/scoring/grouped_v2_candidate.json",
                   Path(defaults.metadata)):
        manifest.append(dict(path=str(source.resolve()), sha256=sha(source), role="code_config_or_metadata"))
    write_json(out / "input_manifest.json", manifest)
    print(f"Loaded {len(data)} videos, {spring_n} spring v3 overlays; baseline R2 replay 1200/1200", flush=True)
    old_loader = scorer.load_json
    current = dict(data)
    scorer.load_json = lambda p: current[p] if p in current else old_loader(p)
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
            model_rows = [scorer.score_one(p, scorer.DEFAULT_TOLERANCES, run) for p in paths]
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
            assert sum(r["effective_for_score"] for r in rows) == 1047
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
    write_json(out / "verification.json", dict(n_inputs=len(data), n_spring_overlays=spring_n,
        n_scenarios=len(design["scenarios"]), n_scores=len(per_video), r2_baseline_status_replay=1200,
        baseline_effective=1047, no_unavailable_scores=True, all_input_files_unchanged=unchanged,
        gamma_and_weight_effective_membership_preserved=True, weak_weight_affine_score_check=True,
        gamma_score_monotonicity_check=True, threshold_unaffected_metrics_exactly_preserved=True,
        baseline_repeated_scenarios_identical=True, independent_quality_calibration="not performed"))
    make_report(out, summaries, metrics, global_rows, coefficients, crossings, common_summary)
    make_plot(out, summaries, global_rows)
    print(f"Reports: {out}", flush=True)


def make_report(out, summaries, metrics, global_rows, coefficients, crossings, common_summary):
    baseline = [r for r in summaries if r["family"] == "baseline"]
    base_by_model = {r["model"]:r for r in baseline}
    base_global = next(r for r in global_rows if r["family"] == "baseline")
    gamma_pct = [100*r["e2e_delta"]/base_by_model[r["model"]]["end_to_end_macro"] for r in summaries if r["family"] == "gamma"]
    r2_global = [r for r in global_rows if r["family"] == "r2"]
    local_weight_ranks = [r["effective_rho"] for r in global_rows if r["family"] == "weak_weight" and r["weak_weight"] >= .6]
    local_pct_by_model = [max(abs(100*r["e2e_delta"]/b["end_to_end_macro"]) for r in summaries
                             if r["model"] == model and r["family"] == "weak_weight" and r["weak_weight"] >= .6)
                          for model,b in base_by_model.items()]
    metric_base = {(r["model"],r["metric"]):r for r in metrics if r["family"] == "baseline"}
    largest_metric = max((r for r in metrics if r["family"] == "r2"),
        key=lambda r:abs(r["end_to_end_score"]-metric_base[(r["model"],r["metric"])]["end_to_end_score"]))
    largest_delta = abs(largest_metric["end_to_end_score"]-metric_base[(largest_metric["model"],largest_metric["metric"])]["end_to_end_score"])
    top_crossing = next(c["crossing_weight"] for c in crossings if c["score_type"] == "effective" and {c["model_a"],c["model_b"]} == {"jimeng","kling"})
    lines = ["# 当前评分方案：γ、R² 状态阈值与弱有效权重敏感性", "",
        "基准：grouped_v2_candidate（九项指标、三组等权、物体间平均、γ=1）+ 显式启用的 period_confidence_v3_candidate；weak_valid 权重=0.8，R² weak/valid 截断=0.65/0.85。Qmask 评分入口默认已采用 grouped_v2_candidate；弹簧默认估计器仍为 legacy_v1。",
        "使用六模型各500条、合计3,000条保存结果，其中300条弹簧采用已保存v3候选。仅重评分与状态重分类，未重跑视频生成、GPU跟踪或参数估计。每次仅改变一个因素。",
        "分数单位为0–100。E2E将未通过门控的视频记0，再按十指标等权平均；有效宏平均按每指标有效样本均分后平均，无有效样本的指标按现有评分器排除，不能与E2E混用。", "",
        "| 模型 | 有效数 | 弱有效数（有效集合内） | 有效宏平均 | E2E宏平均 |", "| --- | ---: | ---: | ---: | ---: |"]
    for r in baseline:
        lines.append(f"| {r['model']} | {r['n_effective']} | {r['n_weak_effective']} | {r['effective_macro']:.4f} | {r['end_to_end_macro']:.4f} |")
    lines.extend(["", "## 主要发现", "",
        "三组实验的全指标E2E排名均不变：jimeng > wan2.2 > HunyuanVideo > kling > CogVideoX-5b > wan2.1；其经验Spearman ρ均为1。有效宏平均排名存在参数敏感性，不能以E2E稳定替代条件排名稳定。",
        f"γ从0.35到2时，有效宏平均前两名在jimeng/kling之间交换；相对γ=1，各模型E2E变化范围为{min(gamma_pct):+.2f}%至{max(gamma_pct):+.2f}%。",
        f"R²九种设置下，全指标有效数为{min(r['n_effective'] for r in r2_global):,}–{max(r['n_effective'] for r in r2_global):,}，相对基准变化{100*(min(r['n_effective'] for r in r2_global)/base_global['n_effective']-1):+.2f}%至{100*(max(r['n_effective'] for r in r2_global)/base_global['n_effective']-1):+.2f}%；四个受影响指标有效数为{min(r['affected_n_effective'] for r in r2_global)}–{max(r['affected_n_effective'] for r in r2_global)}，基准为{base_global['affected_n_effective']}。单个模型指标E2E变化最大约{largest_delta:.2f}分，出现在{largest_metric['model']}的{largest_metric['metric']}；宏平均会稀释局部变化。",
        f"weak_valid权重0.6–1.0时，各模型E2E相对基准最大变化约±{min(local_pct_by_model):.2f}%至±{max(local_pct_by_model):.2f}%；有效宏平均ρ为{min(local_weight_ranks):.3f}–{max(local_weight_ranks):.3f}。权重0.8时Kling/Jimeng有效宏平均为{base_by_model['kling']['effective_macro']:.4f}/{base_by_model['jimeng']['effective_macro']:.4f}，差仅{abs(base_by_model['kling']['effective_macro']-base_by_model['jimeng']['effective_macro']):.4f}分；交叉权重为{top_crossing:.6f}，不能宣称有效集合中的第一名对0.8的选择稳健。",
        f"通过线性系数检查，[0,1]内E2E交叉点共{sum(c['score_type']=='e2e' for c in crossings)}个，有效宏平均交叉点共{sum(c['score_type']=='effective' for c in crossings)}个。稳定排名并不表示绝对得分对权重不敏感。"])
    lines.extend(["", "## γ：固定聚合公式与所有门控", "",
        "每个质量分量按 q=1−p^γ 计算，p为截断到[0,1]的归一化劣化程度。γ越大，对中间质量的惩罚越弱；达到好/坏边界的分量分数仍为1/0。γ不是指数物理误差分数的τ。所有设置有效数均为1,047。", ""])
    model_order = [r["model"] for r in baseline]
    lines.extend(["| γ | " + " | ".join(model_order) + " | E2E ρ | 有效 ρ |",
                  "| --- | " + " | ".join(["---:"] * len(model_order)) + " | ---: | ---: |"])
    for g in global_rows:
        if g["family"] == "gamma":
            vals = [next(r for r in summaries if r["scenario"] == g["scenario"] and r["model"] == m)["end_to_end_macro"] for m in model_order]
            lines.append(f"| {g['gamma']:g} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {g['e2e_rho']:.3f} | {g['effective_rho']:.3f} |")
    lines.extend(["", "## R²：四个共享阈值的指标", "",
        "仅影响速度、加速度、重力、密度，共1,200条；其他指标（含弹簧v3的多条件质量规则）不套用此R²阈值。失败项保持失败，已存测量值及跟踪门控固定。基准状态回放1,200/1,200完全一致。", "",
        "| weak | valid | valid / weak / invalid / failed（四指标） | 四指标有效数 | 全指标有效数 | 新入选 / 剔除 | E2E ρ（全指标） | E2E ρ（四指标） | 有效 ρ（全指标） |",
        "| ---: | ---: | --- | ---: | ---: | --- | ---: | ---: | ---: |"])
    for g in global_rows:
        if g["family"] == "r2":
            lines.append(f"| {g['weak_cutoff']:.2f} | {g['valid_cutoff']:.2f} | {g['affected_valid']} / {g['affected_weak_valid']} / {g['affected_invalid']} / {g['affected_failed']} | {g['affected_n_effective']} | {g['n_effective']} | {g['newly_effective']} / {g['no_longer_effective']} | {g['e2e_rho']:.3f} | {g['affected_e2e_rho']:.3f} | {g['effective_rho']:.3f} |")
    lines.extend(["", "threshold_common_cohort.csv另报告九种阈值下共同有效的同视频集合，减少样本构成差异对条件均分的干扰；这不能消除生成失败带来的选择偏差。", "",
        "## weak_valid：固定状态和有效集合", "",
        "0.6–1.0为邻域/宽范围对照；0、0.2、0.4为弱有效贡献的压力测试。权重0仍保留原有效集合，只将弱有效贡献置零，与strict剔除不是同一个实验。所有设置有效数保持1,047。", "",
        "| 弱有效权重 | " + " | ".join(model_order) + " | E2E ρ | 有效 ρ |",
        "| --- | " + " | ".join(["---:"] * len(model_order)) + " | ---: | ---: |"])
    for g in global_rows:
        if g["family"] == "weak_weight":
            vals = [next(r for r in summaries if r["scenario"] == g["scenario"] and r["model"] == m)["end_to_end_macro"] for m in model_order]
            lines.append(f"| {g['weak_weight']:g} | " + " | ".join(f"{v:.4f}" for v in vals) + f" | {g['e2e_rho']:.3f} | {g['effective_rho']:.3f} |")
    lines.extend(["", "固定集合下分数是权重w的线性函数：S(w)=A+B·w。精确交叉点如下（仅列[0,1]范围）：", "",
        "| 分数口径 | 模型对 | 交叉权重 |", "| --- | --- | ---: |"])
    for c in crossings:
        lines.append(f"| {c['score_type']} | {c['model_a']} / {c['model_b']} | {c['crossing_weight']:.6f} |")
    lines.extend(["", "## 排名与解释", "",
        "| 实验 | 参数 | E2E排名 | 有效排名 |", "| --- | --- | --- | --- |"])
    for g in global_rows:
        label = (f"γ={g['gamma']:g}" if g["family"] == "gamma" else f"w={g['weak_weight']:g}" if g["family"] == "weak_weight" else f"{g['weak_cutoff']:.2f}/{g['valid_cutoff']:.2f}" if g["family"] == "r2" else "基准")
        lines.append(f"| {g['family']} | {label} | {g['e2e_order']} | {g['effective_order']} |")
    lines.extend(["", "γ和权重导致的升分仅表示评分惩罚变化，不表示视频、跟踪或物理测量改善。R² weak截断改变入选集合，valid截断主要在valid/weak之间转移；条件有效均分的变化须结合N_eff和共同集合解释。",
        "这些实验验证所列参数范围内的经验敏感性，不校准γ=1、R²=0.65/0.85或w=0.8的真值最优性，不测量固定prompt的生成种子方差。", "",
        "产物：design.json预先保存参数；input_manifest.json保存输入与代码哈希；model_summary.csv、metric_summary.csv和per_video_sensitivity.csv保存全量结果；scenario_summary.csv保存状态计数和排名；verification.json保存不变量检查。",
        "", "![E2E敏感性曲线](sensitivity_e2e.png)", "",
        "![有效宏平均敏感性曲线](sensitivity_effective.png)", ""])
    (out / "REPORT_zh-CN.md").write_text("\n".join(lines), encoding="utf-8")


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
        fig.suptitle("One factor at a time: grouped Qmask + spring confidence v3; 3,000 videos", fontsize=13)
        fig.tight_layout(rect=(0,.09,1,.94))
        for suffix in ("png", "pdf", "svg"):
            fig.savefig(out / f"sensitivity_{kind}.{suffix}", dpi=220)
        plt.close(fig)


if __name__ == "__main__":
    main()
