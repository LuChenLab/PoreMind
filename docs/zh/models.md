# 支持的模型

[English](../models.md)

PoreMind 支持基于事件特征的分类模型，以及直接处理波形的深度学习模型。

## 传统机器学习

`build_best_model` 使用事件特征和交叉验证评估内置的 scikit-learn 模型。默认特征为 `duration_s`、`blockade_ratio`、`segment_std`、`segment_skew` 和 `segment_kurt`。

```python
best_pkg = analysis.build_best_model(cv=5, scoring="accuracy")
```

## 波形深度学习

`build_DL_model` 支持 `1D-CNN`、`residual-CNN`、`MAGJAM` 和 `MAGJAM_d4`。MAGJAM 直接使用事件波形，不拼接手工计算的事件特征。

```python
# MAGJAM d8（默认）
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

`interp_length` 默认值为 500。`interp_method="interp"` 会对每段波形进行线性插值；`interp_method="padding"` 会在短波形末尾补零，并对长波形从中间截取固定长度。MAGJAM 默认使用 d8，也提供较浅的 d4 版本。

<p align="center"><img src="../_generated/dlmagjam.png" alt="Overview of the MAGJAM framework" width="85%"></p>
<p align="center">Overview of the MAGJAM framework</p>

自定义 DL 特征提取器需要将实例化后的 `torch.nn.Module` 传给 `model`；仅传入类名字符串不能作为模型实例。详见 [DL 模型 API 参考](../functions/build_DL_model.md)。

## 模型保存、加载与预测

模型包辅助函数可保存和加载支持的模型字典，详见 [`save_model_package`](../functions/save_model_package.md) 和 [`load_model_package`](../functions/load_model_package.md)。新样本预测请使用 [`classify_new_samples`](../functions/classify_new_samples.md)。

## 自监督波形表征

当前仓库中的 Python 包尚未导出可调用的 `PoreMindWaveformEncoder` 接口，因此本文档不提供该编码器的运行示例。当前可直接使用上面的手工特征分类或波形分类流程。
