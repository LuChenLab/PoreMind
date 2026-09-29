from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from poremind import __version__, create_analysis_object


def _make_trace(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    sr = 10000
    t = np.arange(0, 1.0, 1 / sr)
    sig = rng.normal(0, 0.3, size=t.shape)
    sig[2000:2060] -= 4.0
    sig[6500:6580] -= 5.0
    pd.DataFrame({"time": t, "current": sig}).to_csv(path, index=False)


def _make_trace_up(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    sr = 10000
    t = np.arange(0, 1.0, 1 / sr)
    sig = rng.normal(-5.0, 0.2, size=t.shape)
    sig[2200:2280] += 2.5
    sig[7000:7080] += 2.0
    pd.DataFrame({"time": t, "current": sig}).to_csv(path, index=False)


def test_object_workflow_end_to_end(tmp_path: Path):
    assert __version__

    s1 = tmp_path / "A1.csv"
    s2 = tmp_path / "B1.csv"
    new_s = tmp_path / "U1.csv"
    _make_trace(s1, 42)
    _make_trace(s2, 7)
    _make_trace(new_s, 100)

    analysis = create_analysis_object(
        sample_paths={"A1": s1, "B1": s2},
        sample_to_group={"A1": "A", "B1": "B"},
        reader="csv",
    ).load()

    assert hasattr(analysis, "pl")
    analysis.denoise(method="moving_average", window=5)
    preview = analysis.preview_signal("A1", start_s=0.0, end_s=0.2)
    assert not preview.empty

    defaults = analysis._default_detect_params("cusum")
    assert {"drift", "threshold", "min_duration_s"}.issubset(defaults.keys())

    analysis.detect_events(
        detect_method="threshold",
        detect_params={"sigma_k": 3.5, "min_duration_s": 0.001},
        baseline_method="rolling_quantile",
        baseline_params={"window": 201, "q": 0.5},
        exclude_current=False,
    )

    feat_df = analysis.extract_features()
    assert len(feat_df) > 0
    assert {"left_baseline", "right_baseline", "blockade_ratio", "channel", "sweep", "sample_id", "segment_skew", "segment_kurt", "peak_factor"}.issubset(feat_df.columns)
    feat_limited = analysis.extract_features(max_event_per_sample=1)
    assert feat_limited.groupby("trace_id").size().max() <= 1
    with pytest.raises(ValueError):
        analysis.extract_features(max_event_per_sample=0)
    feat_df = analysis.extract_features()
    assert len(feat_df) > 0

    analysis.filter_events(
        method="blockade_gmm",
        parameters={"prior_mean": {"A1": 0.2, "B1": 0.2}},
    )
    assert analysis.feature_df is not None
    filtered = analysis.feature_df
    assert "quality_tag" in filtered.columns
    assert analysis.filtered_df is not None
    assert set(analysis.filtered_df["quality_tag"].unique()).issubset({"valid"})
    expected_noise = pd.Series(False, index=analysis.feature_df.index)
    for sample_key, idx in analysis.feature_df.groupby("sample_id").groups.items():
        sub_df = analysis.feature_df.loc[idx].copy()
        valid_mask = analysis._blockade_gmm_mask(
            sub_df,
            rm_index=None,
            blockade_col="blockade_ratio",
            dwell_col="duration_s",
            n_components=2,
            visualize=False,
            prior_mean=0.2,
        )
        expected_noise.loc[idx] = ~valid_mask
    in_lim = (analysis.feature_df["blockade_ratio"].to_numpy(dtype=float) >= 0.1) & (
        analysis.feature_df["blockade_ratio"].to_numpy(dtype=float) <= 1.0
    )
    expected_quality = np.where((~expected_noise.to_numpy()) & in_lim, "valid", "noise")
    assert np.array_equal(filtered["quality_tag"].to_numpy(), expected_quality)
    analysis.filter_events(method="isolation_forest")
    assert analysis.feature_df is not None
    assert "quality_tag" in analysis.feature_df.columns
    analysis.filter_events(method="lof")
    assert analysis.feature_df is not None
    assert "quality_tag" in analysis.feature_df.columns
    analysis.filter_events(
        method="blockade_gmm",
        parameters={"prior_mean": {"A1": 0.2, "B1": 0.2}},
        blockage_lim=(0.0, 2.0),
    )
    assert analysis.filtered_df is not None
    if len(analysis.filtered_df) == 0:
        analysis.filtered_df = analysis.feature_df.copy()
        analysis.filtered_df["is_noise"] = False
        analysis.filtered_df["quality_tag"] = "valid"
    pca_df = analysis.do_pca(feature_cols=["duration_s", "blockade_ratio"], data="filtered")
    assert {"PC1", "PC2"}.issubset(pca_df.columns)
    tsne_df = analysis.do_tsne(feature_cols=["duration_s", "blockade_ratio"], data="filtered", perplexity=5.0, n_iter=300)
    assert {"TSNE1", "TSNE2"}.issubset(tsne_df.columns)
    try:
        umap_df = analysis.do_umap(feature_cols=["duration_s", "blockade_ratio"], data="filtered", n_neighbors=5)
        assert {"UMAP1", "UMAP2"}.issubset(umap_df.columns)
    except ImportError:
        pass

    simple_events = analysis.detect_events_simple(
        sample_id="A1",
        current="denoise",
        start_ms=0.0,
        end_ms=300.0,
        detect_method="threshold",
        detect_params={"sigma_k": 3.0, "min_duration_s": 0.001},
    )
    assert "A1" in simple_events
    assert analysis.detect_events_simple_object == simple_events
    _ = analysis.detect_events_simple(
        sample_id="A1",
        current="raw",
        start_ms=0.0,
        end_ms=300.0,
        detect_method="threshold",
        detect_direction="up",
        baseline_method="global_quantile",
        baseline_params={"q": 0.5},
        detect_params={"sigma_k": 3.0, "min_duration_s": 0.0, "noise_method": "std"},
        merge_event=True,
        merge_event_params={"merge_gap_ms": 0.2},
    )
    _ = analysis.detect_events_simple(
        sample_id="A1",
        current="raw",
        detect_method="threshold",
        exclude_current=False,
    )
    with pytest.raises(ValueError):
        analysis.detect_events_simple(
            sample_id="A1",
            current="raw",
            detect_method="threshold",
            exclude_current=True,
            exclude_current_params={"min": 1e9, "max": None},
        )

    pkg = analysis.build_best_model(cv=2, scoring="accuracy")
    assert "best_model" in pkg
    assert pkg["best_model"] in analysis.model_cv_results
    assert {"duration_s", "blockade_ratio", "segment_std", "segment_skew", "segment_kurt"} == set(pkg["feature_cols"])
    assert "all_samples_feature_pred" in pkg
    assert "best_model_pred" in pkg["all_samples_feature_pred"].columns
    explain = analysis.explain_ml(n_repeats=2)
    assert explain["model_name"] == pkg["best_model"]
    assert explain["method"] == "permutation_importance"
    assert set(explain["ranking"]["feature"]) == set(pkg["feature_cols"])
    assert np.isfinite(explain["ranking"]["importance_mean"].to_numpy(dtype=float)).all()
    assert pkg["best_model"] in analysis.ml_explain_results
    try:
        _ = analysis.pl.model_metric_bar(metric="accuracy", split="test")
        _ = analysis.pl.model_cm(model_name=pkg["best_model"], split="test")
        _ = analysis.pl.plot_2d(data="filtered", value="label")
        _ = analysis.pl.plot_3d(data="filtered", value="label")
        _ = analysis.pl.stacked_bar(group_col="sample_id", value_col="label", data="filtered")
        _ = analysis.pl.box_significance(group_col="label", value_col="blockade_ratio", data="filtered", method="ttest")
        _ = analysis.plot.event_current_simple(sample_id="A1", start_event=1, end_event=2, ylim=(-10, 10))
        _ = analysis.plot.event_current(sample_id="A1", start_event=1, end_event=2, ylim=(-10, 10))
    except ImportError:
        pass

    try:
        import torch  # noqa: F401

        dl_pkg = analysis.build_DL_model(
            model_name="1D-CNN",
            cv=2,
            epoch=2,
            early_stop_patience=1,
            batch_size=16,
            device="cpu",
        )
        assert "model_state_dict" in dl_pkg
        assert analysis.filtered_df is not None
        assert any(c.startswith("pred_label_1D-CNN") for c in analysis.filtered_df.columns)
        _ = analysis.pl.plot_fold_loss(model_name="1D-CNN", type="train")
        try:
            import captum  # noqa: F401

            assert analysis.explain_dl(model_name="1D-CNN", batch_size=8, n_steps=4, device="cpu") is None
            attr = analysis.dl_attribute_results["1D-CNN"]
            assert attr["inputs"].shape == attr["attributions"].shape
            assert attr["inputs"].shape[0] == len(analysis.filtered_df)
            assert {"sample_id", "trace_id", "event_id", "pred_class", "target_class"}.issubset(attr["metadata"].columns)
            _ = analysis.pl.attribute(sample_id="A1", event_index=1, model_name="1D-CNN")
        except ImportError:
            pass
        other_dl, pred_dl = analysis.classify_new_samples({"U1": new_s}, reader="csv", model="1D-CNN")
        assert other_dl.feature_df is not None
        assert any(c.startswith("pred_label_1D-CNN") for c in pred_dl.columns)
    except ImportError:
        pass

    other_analysis, pred = analysis.classify_new_samples(
        {"U1": new_s},
        reader="csv",
        custom_feature_fns={"seg": lambda x: {"ptp": float(np.max(x) - np.min(x))}},
    )
    assert hasattr(other_analysis, "pl")
    assert len(other_analysis.events) > 0
    assert "pred_label" in pred.columns
    proba_cols = [c for c in pred.columns if c.startswith("pred_proba_")]
    assert len(proba_cols) >= 2
    assert "seg_ptp" in pred.columns
    assert other_analysis.feature_df is not None
    assert "pred_label" in other_analysis.feature_df.columns
    for c in proba_cols:
        assert c in other_analysis.feature_df.columns
    try:
        _ = other_analysis.plot.event_current_label(
            sample_id="U1",
            lable_col="pred_label",
            start_event=1,
            end_event=2,
            label_offset=3.0,
            ylim=(-10, 10),
        )
    except ImportError:
        pass


def test_extract_features_up_direction_delta_and_blockade(tmp_path: Path):
    s1 = tmp_path / "UP1.csv"
    _make_trace_up(s1, 123)

    analysis = create_analysis_object(
        sample_paths={"UP1": s1},
        sample_to_group={"UP1": "UP"},
        reader="csv",
    ).load()

    analysis.denoise(method="moving_average", window=5)
    analysis.detect_events(
        detect_method="threshold",
        detect_direction="up",
        detect_params={"sigma_k": 3.0, "min_duration_s": 0.001, "noise_method": "std"},
        baseline_method="global_quantile",
        baseline_params={"q": 0.5},
        exclude_current=False,
    )
    feat_df = analysis.extract_features()
    assert len(feat_df) > 0

    r0 = feat_df.iloc[0]
    expected_delta = -float(r0["global_baseline"]) - (-float(r0["segment_mean"]))
    expected_blockade = expected_delta / (-float(r0["global_baseline"]) + 1e-12)
    assert np.isclose(float(r0["delta_i"]), expected_delta, atol=1e-9)
    assert np.isclose(float(r0["blockade_ratio"]), expected_blockade, atol=1e-9)


def test_explain_errors_before_training(tmp_path: Path):
    s1 = tmp_path / "A1.csv"
    _make_trace(s1, 42)
    analysis = create_analysis_object(
        sample_paths={"A1": s1},
        sample_to_group={"A1": "A"},
        reader="csv",
    ).load()

    with pytest.raises(ValueError, match="build_best_model"):
        analysis.explain_ml()
    with pytest.raises(ValueError, match="model_name"):
        analysis.explain_dl(model_name="missing")
    with pytest.raises(ValueError, match="explain_dl"):
        analysis.pl.attribute(sample_id="A1", event_index=1, model_name="1D-CNN")


def _make_noise_trace(path: Path, seed: int):
    rng = np.random.default_rng(seed)
    sr = 10000
    t = np.arange(0, 1.0, 1 / sr)
    sig = rng.normal(0, 0.3, size=t.shape)
    pd.DataFrame({"time": t, "current": sig}).to_csv(path, index=False)


def test_filter_events_knn_background(tmp_path: Path):
    target_a = tmp_path / "A1.csv"
    target_b = tmp_path / "B1.csv"
    bg = tmp_path / "BG.csv"
    _make_trace(target_a, seed=42)
    _make_trace(target_b, seed=7)
    _make_noise_trace(bg, seed=11)

    analysis = create_analysis_object(
        sample_paths={"A1": target_a, "B1": target_b, "BG": bg},
        sample_to_group={"A1": "A", "B1": "B", "BG": "BG"},
        reader="csv",
    ).load()
    analysis.denoise(method="moving_average", window=5)
    analysis.detect_events(
        detect_method="threshold",
        detect_params={"sigma_k": 3.5, "min_duration_s": 0.001},
        baseline_method="rolling_quantile",
        baseline_params={"window": 201, "q": 0.5},
        exclude_current=False,
    )
    feat_df = analysis.extract_features()
    assert len(feat_df) > 0

    # knn_background with empty background_sample_ids must raise
    with pytest.raises(ValueError, match="background_sample_ids"):
        analysis.filter_events(
            method="knn_background",
            parameters={"background_sample_ids": [], "k": 5, "n_noise_match": 1, "background_ratio": 1.0},
        )

    # Valid config — produces quality_tag and filtered_df (wide blockage_lim
    # because the synthetic trace has near-zero global baseline so blockade_ratio is huge)
    analysis.filter_events(
        method="knn_background",
        parameters={
            "background_sample_ids": ["BG"],
            "k": 5,
            "n_noise_match": 1,
            "background_ratio": 1.0,
            "feature_cols": ["duration_s", "blockade_ratio"],
        },
        blockage_lim=(-1e6, 1e6),
    )
    assert analysis.feature_df is not None
    assert "quality_tag" in analysis.feature_df.columns
    assert analysis.filtered_df is not None
    assert set(analysis.filtered_df["quality_tag"].unique()).issubset({"valid"})

    # background_ratio < 1 should warn but still work; with empty BG the helper
    # warns "background pool empty or background_ratio<=0"
    with pytest.warns(UserWarning, match="background_ratio"):
        analysis.filter_events(
            method="knn_background",
            parameters={
                "background_sample_ids": ["BG"],
                "k": 3,
                "n_noise_match": 99,  # will be clamped to k
                "background_ratio": 0.5,
            },
            blockage_lim=(-1e6, 1e6),
        )
    assert analysis.feature_df is not None
    assert "quality_tag" in analysis.feature_df.columns


def test_knn_background_mask_direct(tmp_path: Path):
    # Direct helper-level test (no full pipeline)
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
    from poremind.workflow import MultiSampleAnalysis
    a = MultiSampleAnalysis.__new__(MultiSampleAnalysis)
    rng = np.random.default_rng(0)
    target = pd.DataFrame({"f1": rng.normal(0, 1, 30), "f2": rng.normal(0, 1, 30)})
    bg_far = pd.DataFrame({"f1": rng.normal(5, 1, 50), "f2": rng.normal(5, 1, 50)})
    bg_near = pd.DataFrame({"f1": rng.normal(0.5, 1, 50), "f2": rng.normal(0.5, 1, 50)})

    # Far background → all valid
    mask_far = a._knn_background_mask(target, bg_far, feature_cols=["f1", "f2"], k=10, n_noise_match=1)
    assert mask_far.sum() == len(target)

    # Near background → most noise
    mask_near = a._knn_background_mask(target, bg_near, feature_cols=["f1", "f2"], k=10, n_noise_match=1)
    assert mask_near.sum() < len(target) // 2

    # Empty background → all valid + warn
    with pytest.warns(UserWarning, match="knn_background"):
        mask_empty = a._knn_background_mask(target, bg_far.iloc[0:0], feature_cols=["f1", "f2"])
        assert mask_empty.sum() == len(target)

    # Clamp n_noise_match to k
    mask_clamp = a._knn_background_mask(target, bg_near, feature_cols=["f1", "f2"], k=5, n_noise_match=999)
    assert len(mask_clamp) == len(target)


def test_filter_events_knn_background_excludes_bg_from_knn(tmp_path: Path):
    """knn_background 时：背景事件 (1) 不参与 KNN 过滤；
    (2) blockage_lim 仍然对背景生效；
    (3) 保留在 feature_df；
    (4) 仅当通过 blockage_lim 才进入 filtered_df。"""
    a1 = tmp_path / "A1.csv"
    bg = tmp_path / "BG.csv"
    _make_trace(a1, seed=42)
    _make_noise_trace(bg, seed=11)

    analysis = create_analysis_object(
        sample_paths={"A1": a1, "BG": bg},
        sample_to_group={"A1": "A", "BG": "BG"},
        reader="csv",
    ).load()
    analysis.denoise(method="moving_average", window=5)
    analysis.detect_events(
        detect_method="threshold",
        detect_params={"sigma_k": 3.5, "min_duration_s": 0.001},
        baseline_method="rolling_quantile",
        baseline_params={"window": 201, "q": 0.5},
        exclude_current=False,
    )
    feat_df = analysis.extract_features()
    assert len(feat_df) > 0

    # 模拟 BG 也有事件被检测出来：手工塞两行 BG 到 feature_df
    bg_rows = pd.concat([feat_df.iloc[0:1], feat_df.iloc[0:1]], ignore_index=True).copy()
    bg_rows["sample_id"] = "BG"
    bg_rows["trace_id"] = "BG__ch0_sw0"
    feat_df_with_bg = pd.concat([feat_df, bg_rows], ignore_index=True)
    # 让其中一行 BG 的 blockade_ratio 在 blockage_lim 内，另一行在范围外
    feat_df_with_bg.loc[feat_df_with_bg.index[-2], "blockade_ratio"] = 0.5   # in lim
    feat_df_with_bg.loc[feat_df_with_bg.index[-1], "blockade_ratio"] = 99.0  # out of lim
    analysis.feature_df = feat_df_with_bg

    analysis.filter_events(
        method="knn_background",
        parameters={
            "background_sample_ids": ["BG"],
            "k": 3,
            "n_noise_match": 1,
            "background_ratio": 1.0,
            "feature_cols": ["duration_s", "blockade_ratio"],
        },
        blockage_lim=(0.1, 1.0),  # 正常区间
    )

    # (1) feature_df 仍包含 BG 行
    assert "BG" in analysis.feature_df["sample_id"].values
    # (2) BG 的 is_noise 取决于 blockage_lim，与其它过滤方法一致
    bg_in = analysis.feature_df[(analysis.feature_df["sample_id"] == "BG") &
 (analysis.feature_df["blockade_ratio"].between(0.1, 1.0))]
    bg_out = analysis.feature_df[(analysis.feature_df["sample_id"] == "BG") &
 (~analysis.feature_df["blockade_ratio"].between(0.1, 1.0))]
    assert len(bg_in) > 0 and len(bg_out) > 0
    assert (bg_in["is_noise"] == False).all()    # 在范围内 → valid
    assert (bg_out["is_noise"] == True).all()    # 范围外 → 被 blockage_lim 硬拒
    assert (bg_in["quality_tag"] == "valid").all()
    assert (bg_out["quality_tag"] == "noise").all()
    # (3) filtered_df 仅包含通过 blockage_lim 的 BG 行
    bg_in_filt = analysis.filtered_df[analysis.filtered_df["sample_id"] == "BG"]
    assert len(bg_in_filt) == len(bg_in)
    assert "BG" not in analysis.filtered_df.loc[
        analysis.filtered_df["blockade_ratio"] == 99.0, "sample_id"
    ].values

