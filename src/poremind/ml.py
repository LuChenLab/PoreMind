from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pickle

import pandas as pd

from .features import select_feature_columns


@dataclass
class LabeledDataset:
    df: pd.DataFrame
    label_col: str = "label"

    def validated(self) -> "LabeledDataset":
        if self.label_col not in self.df.columns:
            raise ValueError(f"missing label column: {self.label_col}")
        if self.df[self.label_col].isna().any():
            raise ValueError("label column contains NaN")
        return self


def _build_model(model_name: str, **kwargs: Any):
    if model_name == "random_forest":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(**kwargs)
    if model_name == "xgboost":
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(**kwargs)
    raise ValueError("unsupported model_name; use random_forest or xgboost")


def train_event_classifier(dataset: LabeledDataset, model_name: str = "random_forest", model_params: dict[str, Any] | None = None):
    ds = dataset.validated()
    features = select_feature_columns(ds.df)
    X = ds.df[features]
    y = ds.df[ds.label_col]

    model = _build_model(model_name, **(model_params or {}))
    model.fit(X, y)

    return {
        "model": model,
        "features": features,
        "label_col": ds.label_col,
        "model_name": model_name,
    }


def predict_events(package: dict[str, Any], df: pd.DataFrame) -> pd.DataFrame:
    X = df[package["features"]]
    out = df.copy()
    out["pred_label"] = package["model"].predict(X)
    if hasattr(package["model"], "predict_proba"):
        proba = package["model"].predict_proba(X)
        out["pred_score_max"] = proba.max(axis=1)
    return out


def save_model_package(package: dict[str, Any], path: str | Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(package, f)


def load_model_package(path: str | Path) -> dict[str, Any]:
    with open(path, "rb") as f:
        return pickle.load(f)
