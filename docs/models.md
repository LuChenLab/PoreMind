# Supported models

[简体中文](zh/models.md)

PoreMind provides classifiers for event-feature tables and waveform-based deep-learning models.

## Classical machine learning

`build_best_model` evaluates the built-in scikit-learn model set using event features and cross-validation. By default, it uses `duration_s`, `blockade_ratio`, `segment_std`, `segment_skew`, and `segment_kurt`.

```python
best_pkg = analysis.build_best_model(cv=5, scoring="accuracy")
```

## Waveform deep learning

`build_DL_model` supports `1D-CNN`, `residual-CNN`, `MAGJAM`, and `MAGJAM_d4`. MAGJAM reads the event waveform directly and disables handcrafted feature concatenation.

```python
# MAGJAM d8 (default)
magjam_d8 = analysis.build_DL_model(
    model_name="MAGJAM",
    interp_length=500,
    interp_method="interp",
    device="cuda",
    cv=5,
)

# MAGJAM d4
magjam_d4 = analysis.build_DL_model(
    model_name="MAGJAM_d4",
    interp_length=500,
    interp_method="interp",
    device="cuda",
    cv=5,
)
```

`interp_length` defaults to 500. `interp_method="interp"` linearly resamples each event waveform; `interp_method="padding"` pads short waveforms with zeros and center-crops long waveforms. MAGJAM's d8 architecture is the default; d4 uses the shallower variant.

<p align="center"><img src="_generated/dlmagjam.png" alt="Overview of the MAGJAM framework" width="85%"></p>
<p align="center">Overview of the MAGJAM framework</p>

For a custom DL extractor, pass an instantiated `torch.nn.Module` as `model`; a string naming a class is not a module instance. See the [DL model API reference](functions/build_DL_model.md).

## Save, reload, and predict

The model-package helpers serialize and load supported model dictionaries. See [`save_model_package`](functions/save_model_package.md) and [`load_model_package`](functions/load_model_package.md). For predictions on new traces, use [`classify_new_samples`](functions/classify_new_samples.md).

## Self-supervised waveform embeddings

The repository's current Python package does not export a callable `PoreMindWaveformEncoder` API. This book therefore does not provide executable embedding examples; use the implemented feature-based or waveform-classification workflows above.
