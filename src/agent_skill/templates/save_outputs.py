"""Utility helpers for PoreMind agent runs.

These helpers are intentionally lightweight and platform-neutral. Agents may copy
or adapt them into generated analysis scripts.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def make_run_dir(base: str | Path = "runs", suffix: str = "poremind") -> Path:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(base) / f"{ts}_{suffix}"
    for sub in [
        "logs",
        "s1_input",
        "s2_preview",
        "s4_event_preview",
        "s6_events",
        "s7_filter",
        "s8_visualization",
        "s9_model",
    ]:
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_json(path: str | Path, obj: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def append_log(path: str | Path, title: str, content: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"\n## {datetime.now().isoformat(timespec='seconds')} {title}\n\n")
        f.write(content.rstrip() + "\n")


def save_ax(ax, path: str | Path, dpi: int = 200) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = ax.figure
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def save_df(df: pd.DataFrame, path: str | Path, index: bool = False) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return str(path)


def make_event_summary(df: pd.DataFrame) -> pd.DataFrame:
    group_col = "sample_id" if "sample_id" in df.columns else ("label" if "label" in df.columns else None)
    if group_col is None or len(df) == 0:
        return pd.DataFrame({"n_events": [len(df)]})
    agg = {"event_id": "count"}
    if "duration_s" in df.columns:
        agg["duration_s"] = "median"
    if "blockade_ratio" in df.columns:
        agg["blockade_ratio"] = "median"
    if "snr" in df.columns:
        agg["snr"] = "median"
    out = df.groupby(group_col).agg(agg).reset_index()
    out = out.rename(columns={"event_id": "n_events"})
    return out
