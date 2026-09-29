# `train_event_classifier`
**Module:** `ml.py`

Trains event classifier from labeled dataframe.

## Parameters
- `dataset` (`LabeledDataset`): Labeled dataset wrapper.
- `model_name` (`str`): Model type: random_forest or xgboost.
- `model_params` (`dict[str, Any] | None`): Model hyperparameters.

## Returns
- `dict model package`
