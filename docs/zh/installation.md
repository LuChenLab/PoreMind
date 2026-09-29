# 安装

[English](../installation.md)

## 环境要求

PoreMind 需要 Python 3.10 或更高版本。从 [PoreMind GitHub 仓库](https://github.com/LuChenLab/PoreMind)下载 ZIP 压缩包并解压，然后在项目根目录打开终端。项目根目录中包含 `pyproject.toml` 和 `src/` 文件夹。

## 安装 PoreMind

```bash
conda create -n poremind python=3.10 -y
conda activate poremind
pip install -e .
```

## 按需安装可选依赖

```bash
pip install pyabf        # 读取 ABF 文件
pip install torch        # 波形深度学习
pip install captum tqdm  # DL 归因分析
pip install umap-learn   # UMAP 降维
pip install xgboost      # XGBoost 模型
```

## 检查安装

```python
from poremind import __version__, create_analysis_object

print(__version__)
```

本地预览文档站时，在项目根目录安装文档依赖并准备 Notebook 和图片：

```bash
pip install -r docs/requirements-docs.txt
python scripts/prepare_docs.py
mkdocs serve
```
