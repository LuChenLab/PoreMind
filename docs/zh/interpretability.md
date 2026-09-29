# 模型可解释性

[English](../interpretability.md)

PoreMind 为传统 ML 模型提供置换特征重要性分析，为波形 DL 模型提供 Integrated Gradients（积分梯度）归因。分析结果反映模型对输入数据的响应，不能据此推断因果关系。

## 传统 ML：置换特征重要性

训练模型后，逐一打乱输入特征，并比较指定评价指标的变化。

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

## 波形 DL：Integrated Gradients

运行前安装 `captum` 和 `tqdm`。`model_name` 需与训练时一致，sample ID 需存在于已保存的归因结果中。

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

`pl.attribute` 中的 `event_index` 从 1 开始；`event_index=1` 对应已保存的 `event_id=0`。先对所选模型运行 `explain_dl`，即可使用相同流程查看 `residual-CNN`、`1D-CNN`、`MAGJAM` 或 `MAGJAM_d4` 的归因结果。
