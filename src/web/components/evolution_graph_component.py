from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import streamlit as st
import streamlit.components.v1 as components


def _declare_component():
    dev_url = str(os.environ.get("EVOLUTION_GRAPH_DEV_URL") or "").strip()
    if dev_url:
        return components.declare_component("evolution_graph", url=dev_url)

    build_dir = Path(__file__).resolve().parent.parent / "frontend" / "evolution_graph" / "build"
    if build_dir.exists():
        return components.declare_component("evolution_graph", path=str(build_dir))

    return None


_COMPONENT = _declare_component()


def evolution_graph(
    payload: Dict[str, Any],
    *,
    height: int = 720,
    initial_frame_idx: int = 0,
    speed_ms: int = 420,
    display_mode: str = "",
    key: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    if _COMPONENT is None:
        st.warning("EvolutionGraph 前端未构建：请先构建 src/web/frontend/evolution_graph")
        return None

    out = _COMPONENT(
        payload=payload,
        height=int(height),
        initialFrameIdx=int(initial_frame_idx),
        speedMs=int(speed_ms),
        displayMode=str(display_mode or ""),
        key=key,
        default=None,
    )
    return out if isinstance(out, dict) else None


__all__ = ["evolution_graph"]
