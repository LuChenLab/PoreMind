# PoreMind<img src="poremind_logo.png" align="right" height="100" />

[English](./README.md) | [中文](./README.zh.md)

PoreMind 是一个用于单分子纳米孔信号分析的 Python 框架，可将 ABF 和 CSV 电流记录整理为格式统一的事件数据。它把信号处理、事件检测、特征提取、质量检查、机器学习、波形深度学习和自监督学习纳入同一套流程。用户可通过 Python API、本地 Web UI 或 PoreMind Agent Skill 分析多个样本，从读取原始轨迹到训练模型、预测新样本中的事件。

**核心功能（Key features）**

- **统一格式的事件数据。** 每个事件记录都包含波形、局部基线、起止时间、样本信息和计算得到的特征。整个分析流程都可以使用同一份事件数据。
- **多种信号处理方法和质量检查。** 可根据噪声水平、电流方向、基线稳定性和事件形态，选择不同的降噪、基线校正和事件检测方法。阻塞率过滤和异常检测可帮助识别或去除质量较差的事件；PCA、t-SNE 和 UMAP 可用于查看事件分布。
- **基于特征和波形的模型。** PoreMind 内置 10 种基于事件特征的传统机器学习模型，也支持直接处理事件波形的 `1D-CNN`、`residual-CNN` 和自定义 PyTorch 模型。**MAGJAM** 是一种多阶段时序模型，可直接从电流波形中学习有助于区分事件类型的特征，并在我们的基准测试任务中取得综合最佳表现。
- **自监督学习。** 这类模型可从电流序列中学习一组概括信号的特征，用于可视化和分类，也可与根据事件波形计算的特征进行比较。
- **三种使用方式。** 可通过 Python API、本地 Web UI 或 PoreMind Agent Skill 运行同一套分析流程。Agent Skill 支持根据自然语言指令调用 Python 分析流程。

## 安装

项目要求 Python 3.10 或更高版本。从 GitHub 下载源码安装：

1. 打开 PoreMind 的 GitHub 仓库，点击 **Code → Download ZIP** 下载压缩包。
2. 解压 ZIP 文件，进入解压后的项目文件夹。
3. 在项目根目录打开终端。该目录应能看到 `pyproject.toml` 和 `src/`。

```bash
conda create -n poremind python=3.10 -y
conda activate poremind
pip install -e .
```

安装相关依赖：

```bash
pip install pyabf       # 读取 ABF 文件
pip install torch       # 深度学习训练与预测
pip install captum tqdm # 查看哪些波形片段影响 DL 预测
pip install umap-learn  # UMAP 降维
pip install xgboost     # XGBoost 支持
```

## 文档

- 项目文档：[PoreMind 在线文档](https://LuChenLab.github.io/PoreMind/)

## 快速开始（API）

输入文件通过 sample ID 与文件路径的字典指定。`sample_to_group` 用于设置每个样本的分组；提取特征时，该分组会写入对应事件的 `label`。

```python
from poremind import create_analysis_object

analysis = create_analysis_object(
    {"std_A_01": "std_A_01.abf", "std_B_01": "std_B_01.abf"},
    sample_to_group={"std_A_01": "A", "std_B_01": "B"},
    reader="abf",
).load()

analysis.denoise()  # Python API 默认：butterworth_filtfilt

# 在局部时间窗口内快速调试事件检测参数。
preview_events = analysis.detect_events_simple(
    detect_method="threshold",
    start_ms=0.0,
    end_ms=1000.0,
)

# 对已加载轨迹完整检测事件，然后提取和过滤特征。
analysis.detect_events(detect_method="threshold")
features = analysis.extract_features()
analysis.filter_events(
    method="blockade_gmm",
    parameters={"n_components": 2, "prior_mean": None},
    blockage_lim=(0.1, 1.0),
)

# 查看事件特征并训练传统 ML 模型。
analysis.do_pca(feature_cols=["duration_s", "blockade_ratio"], data="filtered")
best_pkg = analysis.build_best_model(cv=5, scoring="accuracy")
analysis.pl.model_cm(model_name=best_pkg["best_model"], split="test")

# 可选的 DL 训练；环境未安装 PyTorch 时需先安装。
dl_pkg = analysis.build_DL_model(
    model_name="MAGJAM",       # d8；d4 版本使用 "MAGJAM_d4"
    interp_length=500,
    interp_method="interp",   # 或 "padding"
    device="cuda",            # CUDA 不可用时自动回退到 CPU
    cv=5,
)

# 使用已训练的 MAGJAM 模型包预测新轨迹中的事件。
new_analysis, predictions = analysis.classify_new_samples(
    {"unknown_01": "unknown_01.abf"},
    reader="abf",
    model="MAGJAM",
)
```

### 其他 DL 模型

内置 residual CNN：

```python
residual_pkg = analysis.build_DL_model(
    model=None,
    model_name="residual-CNN",
    device="cuda",
    interp_length=500,
    expand=50,
    scale="mad",
    epoch=30,
    batch_size=64,
    learning_rate=1e-4,
    early_stop_patience=5,
    cv=5,
)
```

自定义 1D-CNN 模型：

```python
import torch
import torch.nn as nn


class one_DCNN(nn.Module):
    def __init__(self, out_dim: int = 10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(32, out_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


cnn_pkg = analysis.build_DL_model(
    model=one_DCNN(),
    model_name="1D-CNN",
    device="cuda",
    interp_length=500,
    expand=50,
    scale="mad",
    epoch=30,
    batch_size=64,
    learning_rate=1e-4,
    early_stop_patience=5,
    cv=5,
)
```

### 模型可解释性分析

PoreMind 为传统机器学习（ML）和深度学习（DL）分类器提供可解释性分析。传统 ML 模型使用置换特征重要性：逐一打乱特征，并观察模型评价指标的变化。排序反映模型在当前分析数据中对各特征的依赖程度，不能据此推断因果关系。先运行 `build_best_model` 训练模型：

```python
ml_explanation = analysis.explain_ml(
    model_name=best_pkg["best_model"],  # 可省略，默认解释当前选出的最佳模型
    data="filtered",
    scoring="accuracy",
    n_repeats=10,
)
print(ml_explanation["ranking"])
ml_explanation["ax"].figure  # 在 Notebook 中显示置换特征重要性图
```

对于直接输入事件波形的 DL 模型，Integrated Gradients（积分梯度）会相对于基线为波形各位置计算归因分数，用于分析哪些波形区域影响模型输出。该归因反映模型对输入的响应，不代表因果效应。运行前需安装 `captum` 和 `tqdm`。`model_name` 应与训练时一致，`std_A_01` 替换为数据中实际存在的 sample ID：

```python
analysis.explain_dl(
    model_name="residual-CNN",
    method="integrated_gradients",
    baseline="zero",
    n_steps=50,
)
analysis.pl.attribute(
    sample_id="std_A_01",
    event_index=1,
    model_name="residual-CNN",
)
```

## Notebook 示例

运行 Notebook 示例：[`notebooks/step_by_step_analysis.ipynb`](./notebooks/step_by_step_analysis.ipynb)。

## 本地 Web UI

在项目环境中启动 Gradio 应用：

```bash
poremind-ui
# 或在源码目录中运行
python -m ui.app
```

启动后会自动在浏览器中打开应用页面，也可以自行访问：

- http://127.0.0.1:7860/

UI 按九个步骤组织：**Import**（导入）、**Preprocess**（预处理）、**Pre-Events**（局部检测调参）、**Events**（完整事件检测）、**Features & Filter**（特征与过滤）、**Reduction**（降维）、**Train Model**（模型训练）、**Predict**（预测）和 **Export**（导出）。

在第 8 步加载已保存的模型，可对训练时未使用的新样本进行分类。

YouTube 上有 PoreMind 本地 Web UI 的演示视频：

[![PoreMind Local Web Demo](https://img.youtube.com/vi/kSs1sbFrdPc/maxresdefault.jpg)](https://youtu.be/kSs1sbFrdPc)

## 使用说明

- **输入数据：** ABF 读取器默认加载全部 channel 和 sweep。CSV 文件需要包含 `current` 列，也可以包含 `time` 列；没有 `time` 列时，需在读取器选项中指定采样率。
- **标签与分组：** 提取特征时，`sample_to_group` 会将样本分组写入该样本各事件的 `label`。未使用 `sample_to_group` 时，监督式训练前需添加 `label` 列。
- **预处理：** 降噪方法包括 `butterworth_filtfilt`、`moving_average`、`median`、`drift_corrected_moving_average` 和 `none`。
- **事件检测：** 支持 `threshold`、`zscore_threshold`、`cusum`、`pelt` 和 `hmm`。Python API 默认使用 `detect_direction="down"`、`baseline_method="rolling_quantile"`（`window=10000`、`q=0.5`）和 `exclude_current=True`。默认不合并相邻事件，也支持 `global_quantile` 和 `global_median` 基线。UI 默认值可能不同；复现分析时需明确设置参数。
- **特征与过滤：** 默认事件特征包括 `duration_s`、`blockade_ratio`、`segment_std`、`segment_skew` 和 `segment_kurt`。过滤方法包括 `blockade_gmm`、`peak_detection`、`isolation_forest`、`lof` 和 `knn_background`。模型过滤前会先按 `blockage_lim=(0.1, 1.0)` 去除阻塞率超出范围的事件。
- **模型：** `build_best_model` 用于比较和训练传统 ML 模型；`build_DL_model` 支持 `residual-CNN`、`MAGJAM` 和 `MAGJAM_d4`。`MAGJAM` 是 d8 版本，两个 MAGJAM 版本都直接使用事件波形，无需预先计算事件特征。
- **DL 波形处理：** `interp_length` 默认是 `500`，`interp_method` 默认是 `"interp"`。`interp` 对波形进行线性重采样；`padding` 在短片段末尾补零，并从长片段中间截取固定长度。其他参数包括 `scale`、`expand`、`batch_size`、`learning_rate`、`epoch` 和 `early_stop_patience`。
- **新样本预测：** `classify_new_samples` 可使用传统 ML 或 DL 模型预测事件类别。ML 结果列为 `pred_label` 和 `pred_proba_<class>`；DL 结果列为 `pred_label_<model_name>` 和 `pred_proba_<class>_<model_name>`。
- **查看模型依据与可选工具：** `analysis.plot` 是 `analysis.pl` 的别名。安装 PyTorch、Captum 和 tqdm 后，可通过 `analysis.explain_dl` 和 `analysis.pl.attribute` 查看哪些波形片段影响 DL 预测。使用 UMAP 和 XGBoost 也需安装前文列出的依赖。

## 许可

PoreMind 根据 [PolyForm Noncommercial License 1.0.0](./LICENSE) 授权使用。该许可允许非商业用途，不限于科研；商业用途需另行取得版权持有者授权。具体条款见许可文件。

## 联系与引用

如有疑问、问题反馈或功能建议，请前往 [GitHub Issues 提交](https://github.com/LuChenLab/PoreMind/issues)。

Defu Liu#, Jing-wen Lin²*, Lu Chen⁎, et al. (2023). *PoreMind: a unified computational framework for single-molecule nanopore signal analysis.* [GitHub 仓库](https://github.com/LuChenLab/PoreMind)。
