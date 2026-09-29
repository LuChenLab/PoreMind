# Model interpretability

[简体中文](zh/interpretability.md)

PoreMind provides permutation feature importance for classical ML models and Integrated Gradients attributions for waveform-based DL models. These methods describe model behavior; they do not establish causal effects.

## Classical ML: permutation feature importance

Train a model first, then permute each input feature and measure the change in the selected score.

```python
ml_explanation = analysis.explain_ml(
    model_name=best_pkg["best_model"],
    data="filtered",
    scoring="accuracy",
    n_repeats=10,
)
print(ml_explanation["ranking"])
ml_explanation["ax"].figure
```

## Waveform DL: Integrated Gradients

Install `captum` and `tqdm` first. Use the same model name as in training and a sample ID present in the stored attribution results.

```python
analysis.explain_dl(
    model_name="MAGJAM",
    method="integrated_gradients",
    baseline="zero",
    n_steps=50,
)
analysis.pl.attribute(
    sample_id="A8",
    event_index=1,
    model_name="MAGJAM",
)
```

`event_index` is one-based in `pl.attribute`; `event_index=1` selects the event whose stored `event_id` is 0. The same workflow supports `residual-CNN`, `1D-CNN`, and both MAGJAM variants after running `explain_dl` for the chosen model.
