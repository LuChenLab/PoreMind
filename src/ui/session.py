from __future__ import annotations

from dataclasses import dataclass, field

from poremind.workflow import MultiSampleAnalysis


@dataclass
class UIAnalysisSession:
    """UI session-level state cache used by the application service layer."""

    analysis: MultiSampleAnalysis | None = None
    sample_paths: dict[str, str] = field(default_factory=dict)
    sample_to_group: dict[str, str] = field(default_factory=dict)

    preprocess_params: dict = field(default_factory=dict)
    detect_params: dict = field(default_factory=dict)
    feature_params: dict = field(default_factory=dict)
    filter_params: dict = field(default_factory=dict)
    model_params: dict = field(default_factory=dict)
    dl_params: dict = field(default_factory=dict)

    outputs: dict = field(default_factory=dict)
