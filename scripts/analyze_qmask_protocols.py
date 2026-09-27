#!/usr/bin/env python3
"""Historical Qmask protocol comparison; current analysis uses grouped_v2_candidate only."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import re
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from physics_eval.quality.protocol import load_protocol, protocol_hash
from scripts.report_io import write_csv, write_json


def import_scorer(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def protocols():
    legacy = load_protocol("legacy_v1")
    grouped = load_protocol("grouped_v2_candidate")
    candidates = [legacy]
    for gamma in (1.0, 1.5, 2.0):
        value = copy.deepcopy(legacy)
        value.update(protocol_id=f"legacy_min_g{gamma:g}", gamma=gamma)
        candidates.append(value)
    for aggregation in ("min", "mean", "group_mean"):
        value = copy.deepcopy(grouped)
        value.update(protocol_id=f"nine_{aggregation}_object_min_g1", aggregation=aggregation, object_aggregation="min")
        candidates.append(value)
    for aggregation in ("mean", "group_mean"):
        value = copy.deepcopy(grouped)
        value.update(protocol_id=f"ten_{aggregation}_object_min_g1", aggregation=aggregation, object_aggregation="min")
        value["components"]["max_centroid_step_norm_by_diameter"] = legacy["components"]["max_centroid_step_norm_by_diameter"]
        value["groups"]["trajectory"] = {"components": ["max_centroid_step_norm_by_diameter"], "weight": 1.0}
        candidates.append(value)
    candidates.append(grouped)
    for gamma in (0.35, 1.5, 2.0):
        value = copy.deepcopy(grouped)
        value.update(protocol_id=f"grouped_mean_g{gamma:g}", gamma=gamma)
        candidates.append(value)
    return candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=ROOT.parent / "results")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline-scorer", type=Path, help="Optional frozen pre-refactor scorer for regression")
    args = parser.parse_args()
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Use a new output directory; previous analyses are immutable")
    out.mkdir(parents=True, exist_ok=True)
    variants = protocols()
    # Freeze candidate rules before reading/scoring model results.
    write_json(out / "protocol_matrix.json", [{"config": p, "hash": protocol_hash(p)} for p in variants])
    scorer = import_scorer(ROOT / "scripts/score_results.py", "qmask_scorer")
    original = import_scorer(args.baseline_scorer.resolve(), "original_scorer") if args.baseline_scorer else None
    saved_argv = sys.argv
    sys.argv = ["score", "--no-plots"]
    try:
        defaults = scorer.parse_args()
    finally:
        sys.argv = saved_argv
    data, manifest = {}, []
    model_paths = {}
    for model in sorted(args.results_root.iterdir()):
        if not model.is_dir():
            continue
        paths = scorer.discover_result_files(model, None)
        if not paths:
            continue
        model_paths[model.name] = paths
        for path in paths:
            raw = path.read_bytes()
            data[path] = json.loads(raw)
            manifest.append({"path": str(path.resolve()), "sha256": hashlib.sha256(raw).hexdigest()})
    if not data:
        raise SystemExit("No saved results found")
    write_json(out / "input_manifest.json", manifest)
    old_loader = scorer.load_json
    scorer.load_json = lambda p: data[p] if p in data else old_loader(p)
    if original:
        original.load_json = scorer.load_json
    summaries, metrics, compact, primary, risks = [], [], [], {}, []
    regression_max = 0.0
    for model, paths in model_paths.items():
        completeness = scorer.check_completeness(paths, defaults.metadata)
        if any(completeness[key] for key in ("missing_ids", "unexpected_ids", "duplicate_ids")):
            raise ValueError(f"Incomplete model: {model}: {completeness}")
        base_rows = None
        for protocol in variants:
            run = copy.copy(defaults)
            run.model_name = model
            run.result_root = str(args.results_root / model)
            run.qmask_protocol = protocol
            rows = [scorer.score_one(path, scorer.load_tolerances(run), run) for path in paths]
            if any(not row["score_available"] for row in rows):
                write_json(out / "incomplete_evidence.json", [r for r in rows if not r["score_available"]])
                raise ValueError("Missing required evidence; refusing incomplete ranking")
            if base_rows is None:
                base_rows = rows
                if original:
                    for path, row in zip(paths, rows):
                        old = original.score_one(path, original.load_tolerances(run), run)
                        regression_max = max(regression_max, abs(old["adjusted_score"] - row["adjusted_score"]))
                        if old["effective_for_score"] != row["effective_for_score"]:
                            raise AssertionError("Legacy effective set changed")
            for before, row in zip(base_rows, rows):
                if (before["effective_for_score"], before["measured_value"], before["base_score"]) != (row["effective_for_score"], row["measured_value"], row["base_score"]):
                    raise AssertionError("Qmask change altered physics or gate")
            summary = scorer.summarize(rows)
            qvalues = [r["mask_quality_score"] for r in rows if r["both_gates_pass"]]
            summaries.append({"model": model, "protocol": protocol["protocol_id"],
                              "n_total": len(rows), "n_effective": summary["n_effective_for_score"],
                              "effective_macro": summary["overall_effective_video_score"],
                              "end_to_end_macro": summary["overall_end_to_end_score"],
                              "mean_qmask_effective": statistics.mean(qvalues) if qvalues else None,
                              "zero_qmask_effective": sum(q == 0 for q in qvalues)})
            metrics.extend({"model": model, "protocol": protocol["protocol_id"], **m} for m in summary["metrics"])
            for path, before, row in zip(paths, base_rows, rows):
                canonical_id = re.search(r"(?:^|[_-])([A-Z]+\d{3})$", str(data[path].get("metadata", {}).get("Prompt_ID") or row["prompt_id"])).group(1)
                compact.append({"model": model, "protocol": protocol["protocol_id"], "metric": row["metric"],
                                "prompt_id": canonical_id, "effective": row["effective_for_score"],
                                "base_score": row["base_score"], "status_multiplier": row["status_multiplier"],
                                "qmask": row["mask_quality_score"], "adjusted_score": row["adjusted_score"],
                                "score_delta_from_legacy": row["adjusted_score"] - before["adjusted_score"]})
                if protocol["protocol_id"] == "grouped_v2_candidate" and row["both_gates_pass"]:
                    details = row["qmask_details"]
                    minimum = min(o["minimum_component"] for o in details["objects"])
                    if minimum == 0 and row["mask_quality_score"] >= 0.8:
                        risks.append({"model": model, "prompt_id": row["prompt_id"], "metric": row["metric"],
                                      "old_qmask": before["mask_quality_score"], "new_qmask": row["mask_quality_score"],
                                      "old_score": before["adjusted_score"], "new_score": row["adjusted_score"],
                                      "zero_components": json.dumps({str(o["object_id"]): [k for k,v in o["components"].items() if v == 0] for o in details["objects"]}),
                                      "source": str(path)})
            if protocol["protocol_id"] in {"legacy_v1", "grouped_v2_candidate"}:
                primary[(model, protocol["protocol_id"])] = rows
                write_json(out / protocol["protocol_id"] / model / "score_summary.json",
                           {"model": model, "scoring": {"qmask_protocol": protocol, "hash": protocol_hash(protocol),
                            "status_multipliers": scorer.status_multipliers_for_report(run), "tolerances": scorer.load_tolerances(run)},
                            "completeness": completeness, "summary": summary, "per_video": rows})
        print(f"Scored {model}: {len(paths)} inputs, {len(variants)} protocols", flush=True)
    if original and regression_max > 1e-10:
        raise AssertionError(f"Legacy regression error: {regression_max}")
    for protocol in variants:
        items = [r for r in summaries if r["protocol"] == protocol["protocol_id"]]
        for rank, item in enumerate(sorted(items, key=lambda r: r["effective_macro"], reverse=True), 1):
            item["rank"] = rank
    write_csv(out / "model_summary.csv", summaries)
    write_csv(out / "metric_summary.csv", metrics)
    write_csv(out / "per_video_sensitivity.csv", compact)
    write_json(out / "high_score_zero_component_review.json", risks)
    write_csv(out / "high_score_zero_component_review.csv", risks)
    # Same prompt IDs effective in every model, reported separately per metric.
    common_rows = []
    for metric in sorted({r["metric"] for r in compact}):
        models = list(model_paths)
        sets = [{r["prompt_id"] for r in compact if r["model"] == model and r["metric"] == metric and r["protocol"] == "legacy_v1" and r["effective"]} for model in models]
        common = set.intersection(*sets)
        for model in models:
            for protocol in ("legacy_v1", "grouped_v2_candidate"):
                vals = [r["adjusted_score"] for r in compact if r["model"] == model and r["metric"] == metric and r["protocol"] == protocol and r["prompt_id"] in common]
                common_rows.append({"metric": metric, "model": model, "protocol": protocol,
                                    "n_common": len(common), "common_effective_score": statistics.mean(vals) if vals else None})
    write_csv(out / "common_effective_subset.csv", common_rows)
    unchanged = all(hashlib.sha256(Path(m["path"]).read_bytes()).hexdigest() == m["sha256"] for m in manifest)
    write_json(out / "verification.json", {"n_inputs": len(data), "n_protocols": len(variants),
               "legacy_original_checked": original is not None, "legacy_max_absolute_score_delta": regression_max if original else None,
               "physics_and_effective_sets_unchanged": True, "input_files_unchanged": unchanged,
               "high_qmask_zero_component_cases": len(risks), "independent_quality_validation": "pending"})
    if not unchanged:
        raise AssertionError("Input files changed during analysis")
    lines = ["# Qmask 新旧协议重评分", "", "仅使用保存诊断，未重跑跟踪；候选协议未经独立人工/真值校准。", "",
             "| 模型 | 有效 N | 旧有效宏平均 | 新有效宏平均 | 变化 | 旧端到端 | 新端到端 | 旧→新排名 |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |"]
    for model in model_paths:
        old = next(r for r in summaries if r["model"] == model and r["protocol"] == "legacy_v1")
        new = next(r for r in summaries if r["model"] == model and r["protocol"] == "grouped_v2_candidate")
        lines.append(f"| {model} | {old['n_effective']}/500 | {old['effective_macro']:.4f} | {new['effective_macro']:.4f} | {new['effective_macro']-old['effective_macro']:+.4f} | {old['end_to_end_macro']:.4f} | {new['end_to_end_macro']:.4f} | {old['rank']} → {new['rank']} |")
    lines.extend(["", f"旧代码回归最大分差：{regression_max if original else '未提供旧代码'}。",
                  f"单项零分但候选 Qmask ≥0.8 的有效样本：{len(risks)}；见复核清单。",
                  "分数上涨来自协议惩罚减弱，不表示模型视频质量改善。有效集合与物理估计保持不变。",
                  "model_summary.csv 给出全部候选及排名，metric_summary.csv 给出分指标数据，per_video_sensitivity.csv 给出逐视频变化。",
                  "候选参数在读取数据前写入 protocol_matrix.json；输入哈希和回归结果保存在 input_manifest.json、verification.json。"])
    (out / "REPORT_zh-CN.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print("Reports:", out, flush=True)


if __name__ == "__main__":
    main()
