# 分析流程

[English](../workflow.md)

Python API 和本地 UI 使用相同的事件分析流程。Pre-Events 在选定的局部时间范围内调试参数，Events 则对已加载的完整信号进行检测。

| 步骤 | 用途 | 主要结果 |
| --- | --- | --- |
| 导入 | 读取 ABF 或 CSV 电流信号并设置样本分组 | 信号轨迹和样本信息 |
| 预处理 | 降噪并查看原始或处理后的信号 | 预处理信号 |
| Pre-Events | 在短时间窗口内调试事件检测参数 | 预览事件和检测参数 |
| Events | 对完整信号检测事件 | 事件和基线信息 |
| 特征与筛选 | 计算事件特征并去除低质量或异常事件 | 特征表和筛选后的事件 |
| 降维 | 使用 PCA、t-SNE 或 UMAP 查看数值特征 | 添加到事件表中的降维坐标 |
| 模型训练 | 训练传统 ML 或波形 DL 模型 | 模型包和评估结果 |
| 预测 | 用训练好的模型分析新样本 | 事件类别和预测分数 |
| 导出 | 保存所选结果和分析参数 | 指定文件夹中的输出文件 |

## 事件数据与标签

每条轨迹都对应一个 sample ID。`sample_to_group` 将样本 ID 映射为实验分组；提取特征时，这些分组会写入事件的 `label` 列。若未提供分组映射，监督式训练前需为数据添加 `label` 列。

传统模型默认使用 `duration_s`、`blockade_ratio`、`segment_std`、`segment_skew` 和 `segment_kurt`。`blockade_ratio` 是无单位比值。常用显示和筛选范围为 0–1；超出范围的值应检查来源，不要直接缩放。

## 事件检测与筛选

Python API 支持 `threshold`、`zscore_threshold`、`cusum`、`pelt` 和 `hmm` 检测方法，以及滚动或全局基线。复现分析时应明确设置参数，因为 UI 与 API 的默认值可能不同。详见[完整事件检测参考](../functions/detect_events.md)和[局部预览检测参考](../functions/detect_events_simple.md)。

筛选时先应用 `blockage_lim`，再运行所选质量控制或异常检测方法，例如 `blockade_gmm`、`peak_detection`、`isolation_forest`、`lof` 或 `knn_background`。
