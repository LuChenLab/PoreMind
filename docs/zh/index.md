<p><img src="../_generated/poremind_logo.png" align="right" width="260" alt="PoreMind"></p>

# PoreMind 文档

[English](../index.md)

PoreMind 是用于单分子纳米孔信号分析的 Python 框架，可将 ABF 和 CSV 离子电流记录整理为统一的事件数据，并通过一致的流程完成信号预处理、事件检测、特征分析、机器学习、波形深度学习、预测和可视化。

## 从这里开始

- 查看[安装说明](installation.md)，安装 PoreMind 及所需的可选依赖。
- 查看 [Quick Start (API)](../_generated/quickstart.ipynb)。
- 了解[分析流程](workflow.md)、[支持的模型](models.md)和[函数参考](api.md)。
- 使用[本地 Web UI](local-ui.md)进行交互式分析。

## 主要功能

- **统一的事件数据。** 事件记录保留波形、时间边界、局部基线、样本信息和事件特征，供后续分析共同使用。
- **多种信号处理方法。** 可选择降噪、基线估计、事件检测和事件质量控制方法；PCA、t-SNE 和 UMAP 可用于查看事件特征分布。
- **传统机器学习和波形深度学习。** 可基于事件特征训练传统模型，或使用 `1D-CNN`、`residual-CNN` 和 **MAGJAM** 直接分析事件波形。MAGJAM 提供 d8 和 d4 两种深度版本。
- **三种使用方式。** 可通过 Python API、本地 Web UI 或 [PoreMind Agent Skill](https://github.com/LuChenLab/PoreMind/blob/main/src/agent_skill/SKILL.md) 使用分析流程。

## 分析流程

```text
ABF / CSV 信号 → 预处理 → 事件检测 → 事件特征与质量控制
               → 可视化 → ML / DL → 新样本预测
```

本手册以当前仓库中的实现为准。[方法框架](../nanopore_single_molecule_framework.md)介绍了现有数据结构、分析方法和功能范围。
