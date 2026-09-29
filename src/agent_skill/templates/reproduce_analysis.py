"""Template: reproduce a locked PoreMind run from 02_params.lock.json."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from poremind import create_analysis_object


def load_params(run_dir: str | Path) -> dict:
    return json.loads((Path(run_dir) / "02_params.lock.json").read_text(encoding="utf-8"))


def reproduce(run_dir: str | Path):
    run_dir = Path(run_dir)
    params = load_params(run_dir)

    inp = params["input"]
    analysis = create_analysis_object(
        sample_paths=inp["sample_paths"],
        sample_to_group=inp.get("sample_to_group"),
        reader=inp.get("reader", "abf"),
        reader_kwargs=inp.get("reader_kwargs"),
    ).load()

    prep = params.get("preprocess", {})
    analysis.denoise(prep.get("method", "butterworth_filtfilt"), **prep.get("params", {}))

    analysis.detect_events(**params["event_detection"])

    feat_params = params.get("feature_extraction", {})
    # Custom feature functions must be restored manually if they were used.
    feature_df = analysis.extract_features(
        custom_feature_fns=None,
        max_event_per_sample=feat_params.get("max_event_per_sample"),
    )
    (run_dir / "s6_events").mkdir(exist_ok=True, parents=True)
    feature_df.to_csv(run_dir / "s6_events/feature_df.reproduced.csv", index=False)

    filtering = params.get("filtering")
    if filtering:
        analysis.filter_events(**filtering)
        if analysis.filtered_df is not None:
            (run_dir / "s7_filter").mkdir(exist_ok=True, parents=True)
            analysis.filtered_df.to_csv(run_dir / "s7_filter/filtered_df.reproduced.csv", index=False)

    modeling = params.get("modeling")
    if modeling and analysis.filtered_df is not None and modeling.get("label_col", "label") in analysis.filtered_df.columns:
        model_pkg = analysis.build_best_model(**modeling)
        (run_dir / "s9_model").mkdir(exist_ok=True, parents=True)
        pd.DataFrame([{"model": k, "score": v} for k, v in model_pkg.get("scores", {}).items()]).to_csv(
            run_dir / "s9_model/ml_model_metrics.reproduced.csv", index=False
        )

    return analysis


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", help="Run directory containing 02_params.lock.json")
    args = parser.parse_args()
    reproduce(args.run_dir)
