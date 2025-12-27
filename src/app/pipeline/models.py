from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class PipelineRunState:
    run_id: str
    project_id: str
    pipeline_name: str
    total_steps: int


@dataclass
class StepState:
    step_idx: int  # 0-based
    step_id: str
    tool: str
    output_key: str
    status: str  # pending|running|success|failed|skipped
    started_at: str
    ended_at: str
    duration_ms: float
    error: Optional[str] = None
    inputs_resolved: Optional[Dict[str, Any]] = None






