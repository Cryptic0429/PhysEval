"""Compact inputs must produce the same execution records as legacy inputs."""
from __future__ import annotations

import copy
import json
from pathlib import Path
import sys

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from physics_eval import estimation, scoring
from physics_eval.io.metadata import load_metadata, read_metadata
from physics_eval.tasks import resolve_task_fields
from tests.test_refactor_equivalence import _fixtures, _score_config, _score_evidence

ROOT = Path(__file__).resolve().parents[1]
LEGACY = pd.read_csv(Path(__file__).parent / "fixtures" / "legacy_metadata.csv")
LEGACY_ROWS = LEGACY.where(pd.notna(LEGACY), None).to_dict(orient="records")
EXECUTION_FIELDS = ("Evaluator", "Measurement_Method")


def compact(row):
    return {key: value for key, value in row.items() if key not in EXECUTION_FIELDS}


@pytest.mark.parametrize("legacy", LEGACY_ROWS, ids=[r["Metric"] for r in LEGACY_ROWS])
def test_legacy_fields_restore_exactly_without_mutating_input(legacy):
    reduced = compact(legacy)
    assert resolve_task_fields(reduced) == legacy
    assert all(key not in reduced for key in EXECUTION_FIELDS)


@pytest.mark.parametrize("explicit", ["custom_evaluator", "", None])
def test_explicit_legacy_values_are_never_replaced(explicit):
    row = dict(Metric="velocity", Evaluator=explicit, Measurement_Method=explicit)
    assert resolve_task_fields(row) == row


def test_partial_defaults_and_unknown_tasks():
    assert resolve_task_fields(dict(Metric="velocity", Evaluator="custom")) == dict(
        Metric="velocity", Evaluator="custom", Measurement_Method="linear_fit",
    )
    for metric in ["tracking_only", "custom_task", ""]:
        row = dict(Metric=metric)
        assert resolve_task_fields(row) == row


@pytest.mark.parametrize("legacy", LEGACY_ROWS, ids=[r["Metric"] for r in LEGACY_ROWS])
@pytest.mark.parametrize("short_track", [False, True])
def test_estimation_and_scoring_match_for_both_formats(legacy, short_track, tmp_path, monkeypatch):
    track = _fixtures()[legacy["Evaluator"]]
    if short_track:
        track = track.iloc[:1]
    monkeypatch.setattr(estimation, "resolve_tracking_json", lambda *args: tmp_path / "tracking.json")
    monkeypatch.setattr(estimation, "normalize_tracking_json", lambda *args, **kwargs: track.copy())
    kwargs = dict(metadata_path=tmp_path / "metadata.xlsx", json_root=None,
                  output_dir=tmp_path, fps=None, save_diag=False, strict=False)
    old = estimation.process_row(legacy, **kwargs)
    new = estimation.process_row(compact(legacy), **kwargs)
    # JSON comparison also handles any identically represented NaN failure values.
    assert json.dumps(new, sort_keys=True) == json.dumps(old, sort_keys=True)
    evidence = _score_evidence()
    evidence["metric"] = legacy["Metric"]
    evidence["physics_result"] = old
    other = copy.deepcopy(evidence)
    other["physics_result"] = new
    score_args = (tmp_path / "result.json", scoring.DEFAULT_TOLERANCES, _score_config())
    assert scoring.score_data(evidence, *score_args) == scoring.score_data(other, *score_args)


def test_release_xlsx_and_csv_resolve_to_the_same_500_records():
    directory = ROOT / "benchmark" / "metadata"
    excel = directory / "phys_t2v_bench_metadata.xlsx"
    csv = directory / "csv" / "phys_t2v_bench_metadata__metadata.csv"
    raw = pd.read_excel(excel)
    assert len(raw) == 500 and len(raw.columns) == 18
    assert all(key not in raw.columns for key in EXECUTION_FIELDS)
    xlsx_rows = load_metadata(excel, ROOT)
    assert xlsx_rows == load_metadata(csv, ROOT)
    assert len({row["Prompt_ID"] for row in xlsx_rows}) == 500
    by_metric = {row["Metric"]: row for row in LEGACY_ROWS}
    for row in xlsx_rows:
        expected = by_metric[row["Metric"]]
        for key in EXECUTION_FIELDS:
            assert row[key] == expected[key]
    strict = read_metadata(excel)
    for key in EXECUTION_FIELDS:
        assert list(strict[key]) == [row[key] for row in xlsx_rows]


@pytest.mark.parametrize("compact_input", [False, True])
def test_readers_preserve_supplied_values_and_legacy_records(tmp_path, compact_input):
    table = LEGACY.copy()
    if compact_input:
        table = table.drop(columns=list(EXECUTION_FIELDS))
    else:
        table.loc[0, "Evaluator"] = "custom_evaluator"
        table.loc[0, "Measurement_Method"] = "custom_method"
    xlsx = tmp_path / "metadata.xlsx"
    table.to_excel(xlsx, index=False)
    rows = load_metadata(xlsx, tmp_path)
    strict = read_metadata(xlsx)
    assert rows[0]["Evaluator"] == ("eval_velocity" if compact_input else "custom_evaluator")
    for key in EXECUTION_FIELDS:
        assert list(strict[key]) == [row[key] for row in rows]
