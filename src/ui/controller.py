from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, BadZipFile, ZipFile

import pandas as pd

from poremind.features import select_feature_columns
from poremind.workflow import MultiSampleAnalysis, create_analysis_object

from .session import UIAnalysisSession


_CLASSICAL_MODEL_BUNDLE_FORMAT = "poremind.classical-model"
_CLASSICAL_MODEL_BUNDLE_VERSION = 1


class AnalysisController:
    """Application service layer that isolates UI from core algorithm internals."""

    def __init__(self, session: UIAnalysisSession | None = None) -> None:
        self.session = session or UIAnalysisSession()

    def load_samples(
        self,
        sample_paths: dict[str, str],
        sample_to_group: dict[str, str] | None = None,
        reader: str = "abf",
        reader_kwargs: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        analysis = create_analysis_object(
            sample_paths=sample_paths,
            sample_to_group=sample_to_group,
            reader=reader,
            reader_kwargs=reader_kwargs or {},
        )
        analysis.load()
        self.session.sample_paths = sample_paths.copy()
        self.session.sample_to_group = (sample_to_group or {}).copy()
        # A newly loaded sample set starts a fresh analysis run.  Do not leave
        # tables or export artifacts from the previous analysis in the session.
        self.session.outputs.clear()
        self.session.preprocess_params = {}
        self.session.detect_params = {}
        self.session.feature_params = {}
        self.session.filter_params = {}
        self.session.model_params = {}
        self.session.dl_params = {}
        self.session.analysis = analysis

        samples = []
        for trace_id, tr in analysis.traces.items():
            sample_id = analysis.trace_to_sample.get(trace_id, trace_id)
            samples.append(
                {
                    "trace_id": trace_id,
                    "sample_id": sample_id,
                    # Keep the import annotation visible under the same column
                    # name consumed by feature extraction, training, and plots.
                    "label": (analysis.sample_to_group or {}).get(sample_id, sample_id),
                    "source": str(tr.source),
                    "channel": int(tr.channel),
                    "sweep": int(tr.sweep),
                    "points": int(len(tr.current)),
                    "duration_s": float(tr.time[-1]) if len(tr.time) else 0.0,
                    "sampling_rate_hz": float(tr.sampling_rate_hz),
                }
            )
        df = pd.DataFrame(samples)
        self.session.outputs["samples"] = df
        return {
            "analysis": analysis,
            "sample_df": df,
            "summary": {
                "n_samples": int(len(sample_paths)),
                "n_traces": int(len(analysis.traces)),
                "reader": reader,
                "trace_ids": list(analysis.traces.keys()),
            },
        }

    def run_denoise(self, method: str = "butterworth_filtfilt", **kwargs: Any) -> dict[str, Any]:
        analysis = self._require_analysis()
        analysis.denoise(method=method, **kwargs)
        self.session.preprocess_params = {"method": method, **kwargs}
        self.session.outputs["denoise"] = {"method": method, "params": kwargs}
        return self.session.outputs["denoise"]

    def run_detect(
        self,
        detect_method: str = "threshold",
        detect_params: dict[str, Any] | None = None,
        baseline_method: str = "rolling_quantile",
        baseline_params: dict[str, Any] | None = None,
        detect_direction: str = "down",
        merge_event: bool = False,
        merge_event_params: dict[str, Any] | None = None,
        exclude_current: bool = True,
        exclude_current_params: dict[str, Any] | None = None,
        stage: str = "global",
        sample_id: str | None = None,
        current: str = "denoise",
        start_ms: float = 0.0,
        end_ms: float = 1000.0,
    ) -> dict[str, Any]:
        analysis = self._require_analysis()
        detect_params = detect_params or analysis._default_detect_params(detect_method)
        baseline_params = baseline_params or {"window": 10000, "q": 0.5}

        if stage == "preview":
            simple = analysis.detect_events_simple(
                detect_method=detect_method,
                detect_params=detect_params,
                baseline_method=baseline_method,
                baseline_params=baseline_params,
                sample_id=sample_id,
                current=current,
                start_ms=start_ms,
                end_ms=end_ms,
                detect_direction=detect_direction,
                merge_event=merge_event,
                merge_event_params=merge_event_params,
                exclude_current=exclude_current,
                exclude_current_params=exclude_current_params,
            )
            out = {"stage": "preview", "event_counts": {k: len(v) for k, v in simple.items()}}
        else:
            analysis.detect_events(
                detect_method=detect_method,
                detect_params=detect_params,
                baseline_method=baseline_method,
                baseline_params=baseline_params,
                detect_direction=detect_direction,
                merge_event=merge_event,
                merge_event_params=merge_event_params,
                exclude_current=exclude_current,
                exclude_current_params=exclude_current_params,
            )
            out = {"stage": "global", "event_counts": {k: len(v) for k, v in analysis.events.items()}}

        self.session.detect_params = {
            "detect_method": detect_method,
            "detect_params": detect_params,
            "baseline_method": baseline_method,
            "baseline_params": baseline_params,
            "detect_direction": detect_direction,
            "merge_event": merge_event,
            "merge_event_params": merge_event_params or {"merge_gap_ms": 0.0},
            "exclude_current": exclude_current,
            "exclude_current_params": exclude_current_params,
        }
        self.session.outputs["detect"] = out
        return out

    def extract_features(
        self,
        max_event_per_sample: int | None = None,
        custom_feature_fns: dict[str, Any] | None = None,
    ) -> pd.DataFrame:
        analysis = self._require_analysis()
        df = analysis.extract_features(max_event_per_sample=max_event_per_sample, custom_feature_fns=custom_feature_fns)
        self.session.feature_params = {"max_event_per_sample": max_event_per_sample, "custom_feature_fns": list((custom_feature_fns or {}).keys())}
        self.session.outputs["feature_df"] = df
        return df

    def filter_events(
        self,
        method: str = "blockade_gmm",
        parameters: dict[str, Any] | None = None,
        blockage_lim: tuple[float, float] = (0.1, 1.0),
    ) -> pd.DataFrame:
        analysis = self._require_analysis()
        analysis.filter_events(method=method, parameters=parameters, blockage_lim=blockage_lim)
        self.session.filter_params = {"method": method, "parameters": parameters or {}, "blockage_lim": blockage_lim}
        assert analysis.filtered_df is not None
        self.session.outputs["filtered_df"] = analysis.filtered_df
        return analysis.filtered_df

    def do_dimensionality_reduction(
        self,
        method: str,
        feature_cols: list[str] | None = None,
        data: str = "filtered",
        **kwargs: Any,
    ) -> pd.DataFrame:
        analysis = self._require_analysis()
        if method == "pca":
            return analysis.do_pca(feature_cols=feature_cols, data=data, **kwargs)
        if method == "tsne":
            return analysis.do_tsne(feature_cols=feature_cols, data=data, **kwargs)
        if method == "umap":
            return analysis.do_umap(feature_cols=feature_cols, data=data, **kwargs)
        raise ValueError("method must be one of: pca, tsne, umap")

    def train_model(
        self,
        label_col: str = "label",
        feature_cols: list[str] | None = None,
        cv: int = 5,
        scoring: str = "accuracy",
        exclude_noise: bool = True,
    ) -> dict[str, Any]:
        analysis = self._require_analysis()
        package = analysis.build_best_model(
            label_col=label_col,
            feature_cols=feature_cols,
            cv=cv,
            scoring=scoring,
            exclude_noise=exclude_noise,
        )
        best_name = str(package["best_model"])
        cv_result = analysis.model_cv_results.get(best_name, {})
        agg = cv_result.get("aggregate", {})
        self.session.model_params = {
            "label_col": label_col,
            "feature_cols": feature_cols,
            "cv": cv,
            "scoring": scoring,
            "exclude_noise": exclude_noise,
        }
        self.session.outputs["model"] = {
            "best_model": best_name,
            "scores": package["scores"],
            "aggregate": agg,
            "feature_cols": package["feature_cols"],
        }
        return self.session.outputs["model"]

    def train_dl_model(
        self,
        model_name: str = "1D-CNN",
        feature_cols: list[str] | None = None,
        interp_length: int = 500,
        interp_method: str = "interp",
        expand: int = 50,
        scale: str | None = "mad",
        device: str = "cpu",
        batch_size: int = 64,
        learning_rate: float = 1e-3,
        epoch: int = 10,
        early_stop_patience: int = 5,
        cv: int = 5,
        label_col: str = "label",
    ) -> dict[str, Any]:
        analysis = self._require_analysis()
        pkg = analysis.build_DL_model(
            model_name=model_name,
            feature_cols=feature_cols,
            interp_length=interp_length,
            interp_method=interp_method,
            expand=expand,
            scale=scale,
            device=device,
            batch_size=batch_size,
            learning_rate=learning_rate,
            epoch=epoch,
            early_stop_patience=early_stop_patience,
            cv=cv,
            label_col=label_col,
        )
        self.session.dl_params = {
            "model_name": model_name,
            "feature_cols": feature_cols,
            "interp_length": interp_length,
            "interp_method": interp_method,
            "expand": expand,
            "scale": scale,
            "device": device,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epoch": epoch,
            "early_stop_patience": early_stop_patience,
            "cv": cv,
            "label_col": label_col,
        }
        self.session.outputs["dl_model"] = {
            "model_name": pkg.get("model_name", model_name),
            "classes": pkg.get("classes", []),
            "feature_cols": pkg.get("feature_cols"),
            "interp_length": pkg.get("interp_length", interp_length),
            "interp_method": pkg.get("interp_method", interp_method),
        }
        return self.session.outputs["dl_model"]

    def predict_new(
        self,
        new_sample_paths: dict[str, str],
        reader: str | None = None,
        reader_kwargs: dict[str, Any] | None = None,
        model: str | None = None,
    ) -> pd.DataFrame:
        analysis = self._require_analysis()
        prediction_analysis, pred = analysis.classify_new_samples(
            new_sample_paths=new_sample_paths,
            reader=reader,
            reader_kwargs=reader_kwargs,
            model=model,
        )
        self.session.outputs["prediction_analysis"] = prediction_analysis
        self.session.outputs["pred_df"] = pred
        return pred

    def save_classical_model_bundle(self, path: str | Path) -> str:
        """Save a trained classical model and its inference configuration as a .pmmodel bundle."""
        analysis = self._require_analysis()
        package = analysis.best_model_package
        if not isinstance(package, dict):
            raise ValueError("Train a classical model before exporting a model bundle.")

        best_name = str(package.get("best_model") or "")
        best_model = package.get("model")
        models = dict(package.get("models") or {})
        if best_name and best_model is not None:
            models.setdefault(best_name, best_model)
        models = {
            str(name): model
            for name, model in models.items()
            if callable(getattr(model, "predict", None))
        }
        if not best_name or best_name not in models:
            raise ValueError("The classical model package does not contain a usable best estimator.")

        feature_cols = [str(col) for col in package.get("feature_cols", [])]
        if not feature_cols:
            raise ValueError("The trained model package has no feature columns to save.")

        payload = {
            "format": _CLASSICAL_MODEL_BUNDLE_FORMAT,
            "version": _CLASSICAL_MODEL_BUNDLE_VERSION,
            "best_model": best_name,
            "models": models,
            "feature_cols": feature_cols,
            "label_col": str(package.get("label_col", "label")),
            "scores": package.get("scores", {}),
            "cv_results": package.get("cv_results") or analysis.model_cv_results,
            "reader": analysis.reader,
            "reader_kwargs": analysis.reader_kwargs,
            "preprocess_state": package.get("preprocess_state", analysis.preprocess_state) or {},
            "detect_state": package.get("detect_state", analysis.detect_state) or {},
            "feature_state": package.get("feature_state", analysis.feature_state) or {},
        }

        try:
            import joblib
        except Exception as exc:  # pragma: no cover - sklearn depends on joblib
            raise ImportError("Saving classical model bundles requires joblib.") from exc

        model_buffer = BytesIO()
        joblib.dump(payload, model_buffer, compress=3)
        manifest = {
            "format": _CLASSICAL_MODEL_BUNDLE_FORMAT,
            "version": _CLASSICAL_MODEL_BUNDLE_VERSION,
            "model_name": best_name,
            "available_models": sorted(models),
            "feature_cols": feature_cols,
            "label_col": payload["label_col"],
            "reader": payload["reader"],
            "inference_only": True,
        }

        output_path = Path(path).expanduser()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        archive_buffer = BytesIO()
        with ZipFile(archive_buffer, "w", compression=ZIP_DEFLATED) as archive:
            archive.writestr("manifest.json", json.dumps(manifest, indent=2, ensure_ascii=False))
            archive.writestr("model.joblib", model_buffer.getvalue())
        output_path.write_bytes(archive_buffer.getvalue())
        return str(output_path.resolve())

    def load_classical_model_bundle(self, path: str | Path) -> dict[str, Any]:
        """Register a trusted .pmmodel bundle for prediction in the current analysis session."""
        bundle_path = Path(path).expanduser()
        if not bundle_path.is_file():
            raise FileNotFoundError(f"Model bundle not found: {bundle_path}")

        try:
            with ZipFile(bundle_path, "r") as archive:
                names = set(archive.namelist())
                if not {"manifest.json", "model.joblib"}.issubset(names):
                    raise ValueError("This file is not a valid PoreMind model bundle.")
                manifest = json.loads(archive.read("manifest.json").decode("utf-8"))
                serialized_model = archive.read("model.joblib")
        except BadZipFile as exc:
            raise ValueError("The selected file is not a valid .pmmodel archive.") from exc

        if manifest.get("format") != _CLASSICAL_MODEL_BUNDLE_FORMAT:
            raise ValueError("Unsupported model bundle format.")
        if manifest.get("version") != _CLASSICAL_MODEL_BUNDLE_VERSION:
            raise ValueError(f"Unsupported PoreMind model bundle version: {manifest.get('version')}")

        # joblib/pickle can execute code while loading. The UI warns users to load
        # only bundles from trusted sources; do not treat this as a safe untrusted format.
        try:
            import joblib
        except Exception as exc:  # pragma: no cover - sklearn depends on joblib
            raise ImportError("Loading classical model bundles requires joblib.") from exc
        payload = joblib.load(BytesIO(serialized_model))

        if not isinstance(payload, dict) or payload.get("format") != _CLASSICAL_MODEL_BUNDLE_FORMAT:
            raise ValueError("The model bundle contents are invalid.")
        if payload.get("version") != _CLASSICAL_MODEL_BUNDLE_VERSION:
            raise ValueError(f"Unsupported model bundle version: {payload.get('version')}")
        models = payload.get("models")
        feature_cols = payload.get("feature_cols")
        best_name = str(payload.get("best_model") or "")
        if not isinstance(models, dict) or not best_name or best_name not in models:
            raise ValueError("The model bundle is missing its best classical estimator.")
        models = {
            str(name): model
            for name, model in models.items()
            if callable(getattr(model, "predict", None))
        }
        if best_name not in models:
            raise ValueError("The best estimator in this bundle cannot perform classical prediction.")
        if not isinstance(feature_cols, list) or not feature_cols or not all(isinstance(col, str) for col in feature_cols):
            raise ValueError("The model bundle has invalid feature column metadata.")

        reader = str(payload.get("reader") or "abf")
        reader_kwargs = payload.get("reader_kwargs") or {}
        if not isinstance(reader_kwargs, dict):
            raise ValueError("The model bundle has invalid reader settings.")

        analysis = self.session.analysis
        if analysis is None:
            analysis = create_analysis_object(
                sample_paths={},
                reader=reader,
                reader_kwargs=reader_kwargs,
            )
            self.session.analysis = analysis
        analysis.reader = reader
        analysis.reader_kwargs = reader_kwargs
        analysis.preprocess_state = dict(payload.get("preprocess_state") or {})
        analysis.detect_state = dict(payload.get("detect_state") or {})
        analysis.feature_state = dict(payload.get("feature_state") or {})
        analysis.model_cv_results = dict(payload.get("cv_results") or {})
        analysis.best_model_package = {
            "model": models[best_name],
            "models": models,
            "best_model": best_name,
            "feature_cols": list(feature_cols),
            "label_col": str(payload.get("label_col") or "label"),
            "scores": dict(payload.get("scores") or {}),
            "cv_results": analysis.model_cv_results,
            "all_samples_feature_pred": pd.DataFrame(),
            "preprocess_state": analysis.preprocess_state,
            "detect_state": analysis.detect_state,
            "feature_state": analysis.feature_state,
        }

        result = {
            "loaded": True,
            "model_name": best_name,
            "available_models": sorted(models),
            "feature_cols": list(feature_cols),
            "label_col": str(payload.get("label_col") or "label"),
            "reader": reader,
            "bundle_path": str(bundle_path.resolve()),
        }
        self.session.outputs["loaded_classical_model"] = result
        return result

    def simple_events_df(self, sample_id: str | None = None) -> pd.DataFrame:
        analysis = self._require_analysis()
        if not analysis.simple_events:
            return pd.DataFrame()
        sid = sample_id or next(iter(analysis.simple_events.keys()))
        rows = [self._event_row(e, sid, i) for i, e in enumerate(analysis.simple_events.get(sid, []))]
        return pd.DataFrame(rows)

    def events_df(self, sample_id: str | None = None) -> pd.DataFrame:
        analysis = self._require_analysis()
        if not analysis.events:
            return pd.DataFrame()
        sid = sample_id or next(iter(analysis.events.keys()))
        rows = [self._event_row(e, sid, i) for i, e in enumerate(analysis.events.get(sid, []))]
        return pd.DataFrame(rows)

    def plot_current(self, sample_id: str | None = None, current: str = "denoise", start_ms: float = 0.0, end_ms: float = 1.0):
        analysis = self._require_analysis()
        ax = analysis.pl.current(sample_id=sample_id, current=current, start_ms=start_ms, end_ms=end_ms)
        return ax.figure

    def plot_event_current_simple(self, sample_id: str | None = None, current: str = "denoise", start_event: int = 1, end_event: int = 5):
        analysis = self._require_analysis()
        ax = analysis.pl.event_current_simple(sample_id=sample_id, current=current, start_event=start_event, end_event=end_event)
        return ax.figure

    def plot_event_current(self, sample_id: str | None = None, current: str = "denoise", start_event: int = 1, end_event: int = 5):
        analysis = self._require_analysis()
        ax = analysis.pl.event_current(sample_id=sample_id, current=current, start_event=start_event, end_event=end_event)
        return ax.figure

    def plot_2d(self, **kwargs: Any):
        analysis = self._require_analysis()
        ax = analysis.pl.plot_2d(**kwargs)
        return ax.figure

    def plot_3d(self, **kwargs: Any):
        analysis = self._require_analysis()
        ax = analysis.pl.plot_3d(**kwargs)
        return ax.figure

    def box_significance(self, **kwargs: Any):
        analysis = self._require_analysis()
        ax = analysis.pl.box_significance(**kwargs)
        return ax.figure

    def plot_model_cm(self, model_name: str, split: str = "test"):
        analysis = self._require_analysis()
        ax = analysis.pl.model_cm(model_name=model_name, split=split)
        return ax.figure

    def plot_model_metric_bar(self, metric: str = "accuracy", split: str = "test"):
        analysis = self._require_analysis()
        ax = analysis.pl.model_metric_bar(metric=metric, split=split)
        return ax.figure

    def plot_fold_loss(self, model_name: str = "1D-CNN", type: str = "train"):
        analysis = self._require_analysis()
        ax = analysis.pl.plot_fold_loss(model_name=model_name, type=type)
        return ax.figure

    def plot_event_current_label(
        self,
        sample_id: str | None = None,
        current: str = "denoise",
        start_event: int = 1,
        end_event: int = 5,
        label_col: str = "pred_label",
    ):
        analysis = self._require_analysis()
        ax = analysis.pl.event_current_label(
            sample_id=sample_id,
            current=current,
            start_event=start_event,
            end_event=end_event,
            lable_col=label_col,
        )
        return ax.figure

    def plot_stacked_bar(self, group_col: str = "sample_id", value_col: str = "pred_label", data: str = "filtered"):
        analysis = self._require_analysis()
        ax = analysis.pl.stacked_bar(group_col=group_col, value_col=value_col, data=data)
        return ax.figure

    def model_prediction_table(self) -> pd.DataFrame:
        analysis = self._require_analysis()
        if analysis.best_model_package is None:
            return pd.DataFrame()
        return analysis.best_model_package.get("all_samples_feature_pred", pd.DataFrame()).copy()

    def feature_table(self) -> pd.DataFrame:
        analysis = self._require_analysis()
        return analysis.feature_df.copy() if analysis.feature_df is not None else pd.DataFrame()

    def filtered_table(self) -> pd.DataFrame:
        analysis = self._require_analysis()
        return analysis.filtered_df.copy() if analysis.filtered_df is not None else pd.DataFrame()

    def export_tables(self, output_dir: str | Path) -> dict[str, str]:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        exported: dict[str, str] = {}
        for name in ["feature_df", "filtered_df", "pred_df"]:
            df = self.session.outputs.get(name)
            if isinstance(df, pd.DataFrame):
                p = out_dir / f"{name}.csv"
                df.to_csv(p, index=False)
                exported[name] = str(p)
        return exported

    def export_params_json(self, path: str | Path) -> str:
        payload = {
            "sample_paths": self.session.sample_paths,
            "sample_to_group": self.session.sample_to_group,
            "preprocess_params": self.session.preprocess_params,
            "detect_params": self.session.detect_params,
            "feature_params": self.session.feature_params,
            "filter_params": self.session.filter_params,
            "model_params": self.session.model_params,
            "dl_params": self.session.dl_params,
        }
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        return str(p)

    def export_analysis_script(self, path: str | Path) -> str:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        script = f'''from poremind.workflow import create_analysis_object

sample_paths = {self.session.sample_paths!r}
sample_to_group = {self.session.sample_to_group!r}

analysis = create_analysis_object(
    sample_paths=sample_paths,
    sample_to_group=sample_to_group,
    reader={self._reader()!r},
)
analysis.load()
analysis.denoise(**{self.session.preprocess_params!r})
analysis.detect_events(**{self.session.detect_params!r})
analysis.extract_features(**{self.session.feature_params!r})
analysis.filter_events(**{self.session.filter_params!r})
'''
        if self.session.model_params:
            script += f'''analysis.build_best_model(**{self.session.model_params!r})
'''
        if self.session.dl_params:
            script += f'''analysis.build_DL_model(**{self.session.dl_params!r})
'''
        p.write_text(script, encoding="utf-8")
        return str(p)

    def suggest_feature_columns(self) -> list[str]:
        analysis = self._require_analysis()
        if analysis.filtered_df is not None:
            return select_feature_columns(analysis.filtered_df)
        if analysis.feature_df is not None:
            return select_feature_columns(analysis.feature_df)
        return []

    def trace_ids(self) -> list[str]:
        analysis = self._require_analysis()
        return list(analysis.traces.keys())

    def _require_analysis(self) -> MultiSampleAnalysis:
        if self.session.analysis is None:
            raise ValueError("Please load samples first.")
        return self.session.analysis

    @staticmethod
    def _event_row(event: Any, sample_id: str, event_id: int) -> dict[str, Any]:
        return {
            "sample_id": sample_id,
            "event_id": int(event_id),
            "start_idx": int(event.start_idx),
            "end_idx": int(event.end_idx),
            "baseline_local": float(event.baseline_local),
            "delta_i": float(event.delta_i),
            "dwell_time_s": float(event.dwell_time_s),
            "snr": float(event.snr),
        }

    def _reader(self) -> str:
        analysis = self._require_analysis()
        return analysis.reader
