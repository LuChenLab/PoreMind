# Installation

[简体中文](zh/installation.md)

## Requirements

PoreMind requires Python 3.10 or later. Download the ZIP archive from the [PoreMind GitHub repository](https://github.com/LuChenLab/PoreMind), extract it, and open a terminal in the extracted project root—the directory containing `pyproject.toml` and `src/`.

## Install the package

```bash
conda create -n poremind python=3.10 -y
conda activate poremind
pip install -e .
```

## Optional dependencies

Install only the packages needed for the planned analysis:

```bash
pip install pyabf        # ABF input
pip install torch        # waveform deep learning
pip install captum tqdm  # DL attribution
pip install umap-learn   # UMAP
pip install xgboost      # XGBoost models
```

## Verify the installation

```python
from poremind import __version__, create_analysis_object

print(__version__)
```

To build or preview this documentation locally, install its separate requirements with `pip install -r docs/requirements-docs.txt`, then run `python scripts/prepare_docs.py` and `mkdocs serve` from the project root.
