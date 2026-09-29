from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


torch = pytest.importorskip("torch")

from poremind.magjam import MAGJAM_MODELS  # noqa: E402
from poremind.workflow import MultiSampleAnalysis  # noqa: E402


@pytest.mark.parametrize(
    ("name", "expected_dim"),
    [("MAGJAM", 8 * 128), ("MAGJAM_d4", 4 * 128)],
)
def test_magjam_registry_and_feature_shapes(name: str, expected_dim: int):
    model = MAGJAM_MODELS[name](n_classes=3, T=32)
    model.eval()
    x = torch.randn(2, 1, 32)
    with torch.no_grad():
        features = model.forward_features(x)
        logits = model(x)
    assert tuple(features.shape) == (2, expected_dim)
    assert tuple(logits.shape) == (2, 3)
    assert model.feat_dim == expected_dim


def _input_analysis() -> MultiSampleAnalysis:
    analysis = MultiSampleAnalysis(sample_paths={})
    analysis.denoised["trace-1"] = np.linspace(-1.0, 1.0, 128, dtype=float)
    return analysis


def test_dl_padding_and_interp_methods():
    analysis = _input_analysis()
    df = pd.DataFrame(
        {
            "trace_id": ["trace-1", "trace-1"],
            "start_idx": [10, 60],
            "end_idx": [20, 80],
        }
    )

    padded, no_features = analysis._build_dl_inputs(
        df,
        interp_length=32,
        expand=0,
        scale=None,
        feature_cols=None,
        interp_method="padding",
    )
    interpolated, _ = analysis._build_dl_inputs(
        df,
        interp_length=32,
        expand=0,
        scale=None,
        feature_cols=None,
        interp_method="interp",
    )
    assert padded.shape == (2, 32)
    assert interpolated.shape == (2, 32)
    assert no_features is None
    assert np.allclose(padded[0, 10:], 0.0)

    with pytest.raises(ValueError, match="interp_method"):
        analysis._build_dl_inputs(
            df,
            interp_length=32,
            expand=0,
            scale=None,
            feature_cols=None,
            interp_method="invalid",
        )


def test_magjam_build_dl_model_uses_waveform_only():
    analysis = _input_analysis()
    rows = []
    for i in range(30):
        start = i * 3
        rows.append(
            {
                "trace_id": "trace-1",
                "start_idx": start,
                "end_idx": start + 2,
                "label": "A" if i % 2 == 0 else "B",
            }
        )
    analysis.filtered_df = pd.DataFrame(rows)

    package = analysis.build_DL_model(
        model_name="MAGJAM_d4",
        interp_length=16,
        interp_method="padding",
        expand=0,
        scale=None,
        device="cpu",
        batch_size=8,
        epoch=1,
        early_stop_patience=1,
        cv=2,
    )

    assert package["model_name"] == "MAGJAM_d4"
    assert package["feature_cols"] is None
    assert package["interp_length"] == 16
    assert package["interp_method"] == "padding"
    assert package["model"]["head"][0].normalized_shape == (4 * 128,)
    assert any(c.startswith("pred_label_MAGJAM_d4") for c in analysis.filtered_df.columns)
