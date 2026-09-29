from __future__ import annotations

import numpy as np
import pandas as pd

from ui.controller import AnalysisController
from ui.app import _default_feature_columns, _parse_sample_annotations


def _make_csv(path, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = 3000
    sr = 1000.0
    t = np.arange(n) / sr
    x = 1.0 + rng.normal(0, 0.01, size=n)
    for s, e in [(400, 430), (900, 940), (1700, 1740), (2400, 2450)]:
        x[s:e] -= 0.5
    pd.DataFrame({"time": t, "current": x}).to_csv(path, index=False)


def test_ui_feature_defaults_and_sample_annotations():
    features = pd.DataFrame(
        {
            "duration_s": [0.1],
            "blockade_ratio": [0.2],
            "segment_std": [0.3],
            "segment_skew": [0.4],
            "segment_kurt": [0.5],
            "peak_factor": [0.6],
            "label": ["old"],
        }
    )
    assert _default_feature_columns(features) == [
        "duration_s",
        "blockade_ratio",
        "segment_std",
        "segment_skew",
        "segment_kurt",
    ]

    annotations = pd.DataFrame(
        {"sample_id": ["sample_a", "sample_b"], "group annotation": ["Class A", float("nan")]}
    )
    assert _parse_sample_annotations(
        annotations,
        {"sample_a": "a.csv", "sample_b": "b.csv"},
    ) == {"sample_a": "Class A", "sample_b": "sample_b"}


def test_controller_end_to_end(tmp_path):
    p1 = tmp_path / "sample_a.csv"
    p2 = tmp_path / "sample_b.csv"
    _make_csv(p1, seed=1)
    _make_csv(p2, seed=2)

    ctl = AnalysisController()
    load_out = ctl.load_samples(
        sample_paths={"sample_a": str(p1), "sample_b": str(p2)},
        sample_to_group={"sample_a": "A", "sample_b": "B"},
        reader="csv",
    )
    assert load_out["summary"]["n_samples"] == 2
    sample_labels = load_out["sample_df"].set_index("sample_id")["label"].to_dict()
    assert sample_labels == {"sample_a": "A", "sample_b": "B"}

    ctl.run_denoise(method="moving_average", window=3)
    prev = ctl.run_detect(
        stage="preview",
        detect_method="threshold",
        detect_params={"sigma_k": 3.0, "min_duration_s": 0.005, "noise_method": "mad"},
        baseline_method="rolling_quantile",
        baseline_params={"window": 101, "q": 0.5},
        sample_id="sample_a",
        start_ms=0,
        end_ms=2000,
    )
    assert prev["stage"] == "preview"

    full = ctl.run_detect(
        stage="global",
        detect_method="threshold",
        detect_params={"sigma_k": 3.0, "min_duration_s": 0.005, "noise_method": "mad"},
        baseline_method="rolling_quantile",
        baseline_params={"window": 101, "q": 0.5},
    )
    assert full["stage"] == "global"

    feature_df = ctl.extract_features()
    assert len(feature_df) > 0
    assert set(feature_df.loc[feature_df["sample_id"] == "sample_a", "label"]) == {"A"}
    assert set(feature_df.loc[feature_df["sample_id"] == "sample_b", "label"]) == {"B"}

    filtered_df = ctl.filter_events(method="blockade_gmm", parameters={"n_components": 2})
    assert "quality_tag" in filtered_df.columns

    model_out = ctl.train_model(cv=2, scoring="accuracy")
    assert "best_model" in model_out

    pred_csv = tmp_path / "unknown.csv"
    _make_csv(pred_csv, seed=3)
    pred_df = ctl.predict_new(new_sample_paths={"unknown": str(pred_csv)}, reader="csv")
    assert len(pred_df) > 0

    model_bundle = ctl.save_classical_model_bundle(tmp_path / "exports" / "classical_model.pmmodel")
    restored_ctl = AnalysisController()
    restored_info = restored_ctl.load_classical_model_bundle(model_bundle)
    assert restored_info["model_name"] == model_out["best_model"]
    assert restored_info["feature_cols"]
    restored_pred = restored_ctl.predict_new(new_sample_paths={"unknown": str(pred_csv)})
    assert len(restored_pred) > 0
    assert "pred_label" in restored_pred.columns

    exported = ctl.export_tables(tmp_path / "exports")
    assert "feature_df" in exported
    params_json = ctl.export_params_json(tmp_path / "exports" / "params_snapshot.json")

    # plotting + event tables
    simple_table = ctl.simple_events_df("sample_a")
    event_table = ctl.events_df("sample_a")
    assert isinstance(simple_table, pd.DataFrame)
    assert isinstance(event_table, pd.DataFrame)

    fig_current = ctl.plot_current(sample_id="sample_a", start_ms=0, end_ms=200)
    fig_simple = ctl.plot_event_current_simple(sample_id="sample_a", start_event=1, end_event=2)
    fig_detect = ctl.plot_event_current(sample_id="sample_a", start_event=1, end_event=2)
    assert fig_current is not None
    assert fig_simple is not None
    assert fig_detect is not None
    script = ctl.export_analysis_script(tmp_path / "exports" / "reproduce_analysis.py")
    assert params_json.endswith("params_snapshot.json")
    assert script.endswith("reproduce_analysis.py")

    ctl.session.outputs["stale_feature_df"] = pd.DataFrame({"label": ["stale"]})
    ctl.session.model_params = {"label_col": "stale"}
    reloaded = ctl.load_samples(
        sample_paths={"sample_a": str(p1), "sample_b": str(p2)},
        sample_to_group={"sample_a": "A2", "sample_b": "B2"},
        reader="csv",
    )
    assert set(ctl.session.outputs) == {"samples"}
    assert ctl.session.model_params == {}
    assert set(reloaded["sample_df"]["label"]) == {"A2", "B2"}
