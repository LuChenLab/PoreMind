# Python API 快速开始

[English](../quickstart.md)

下面的示例展示带分组标签的多样本分析流程。将文件路径、样本 ID 和分组名称替换为实际数据。CSV 文件需要包含 `current` 列；也可以包含 `time` 列。没有时间列时，需要在读取选项中提供采样率。

```python
from poremind import create_analysis_object

analysis = create_analysis_object(
    {
        "sample_a": "path/to/sample_a.abf",
        "sample_b": "path/to/sample_b.abf",
    },
    sample_to_group={"sample_a": "group_a", "sample_b": "group_b"},
    reader="abf",
).load()

# 降噪并查看信号。
analysis.denoise()

# 在短时间窗口内调试检测参数，再对完整信号检测事件。
preview_events = analysis.detect_events_simple(
    detect_method="threshold",
    start_ms=0.0,
    end_ms=1000.0,
)
analysis.detect_events(detect_method="threshold")

# 提取事件特征并筛选事件。
features = analysis.extract_features()
analysis.filter_events(
    method="blockade_gmm",
    parameters={"n_components": 2, "prior_mean": None},
    blockage_lim=(0.1, 1.0),
)

# 查看特征并训练传统分类模型。
analysis.do_pca(
    feature_cols=["duration_s", "blockade_ratio"],
    data="filtered",
)
best_pkg = analysis.build_best_model(cv=5, scoring="accuracy")

# 对训练中未使用的新样本预测事件类别。
new_analysis, predictions = analysis.classify_new_samples(
    {"unknown_01": "path/to/unknown_01.abf"},
    reader="abf",
    model=best_pkg["best_model"],
)
```

`sample_to_group` 会将每个样本对应的分组写入提取后的事件 `label` 列。监督式训练前需提供分组标签。[Notebook 示例](../_generated/quickstart.ipynb)包含当前分析流程的代码和已保存结果。
