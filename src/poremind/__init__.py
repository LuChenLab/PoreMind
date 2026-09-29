"""PoreMind single-molecule nanopore analysis API."""

from .ml import LabeledDataset, predict_events, train_event_classifier
from .pipeline import AnalysisConfig, analyze_abf_to_event_df
from .workflow import MultiSampleAnalysis, create_analysis_object

__version__ = "0.1.0"

_MAGJAM_EXPORTS = {
    "MAGJAM_MODELS",
    "MSSJambaHybrid",
    "MAGJAMExtractor",
    "MAGJAMFullModel",
}


def __getattr__(name: str):
    """Lazily expose MAGJAM symbols without making torch a core import."""
    if name in _MAGJAM_EXPORTS:
        from . import magjam

        return getattr(magjam, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "__version__",
    "AnalysisConfig",
    "analyze_abf_to_event_df",
    "LabeledDataset",
    "train_event_classifier",
    "predict_events",
    "MultiSampleAnalysis",
    "create_analysis_object",
    "MAGJAM_MODELS",
    "MSSJambaHybrid",
    "MAGJAMExtractor",
    "MAGJAMFullModel",
]
