"""CONTEXT COMPACTION node (layer 2 of the memory stack):
Runs after all parallel subagents finish. Summarizes their raw findings into
state.summary so the synthesizer receives a compact, context-window-safe
input instead of unbounded raw text. The raw findings list itself is left in
state for the verify_citations node (which runs after synthesize) and is
cleared there.
"""
from __future__ import annotations

from engine.memory.compaction import compact_findings
from engine.models import LEAD_MODEL
from engine.state import ResearchState


def compact(state: ResearchState) -> dict[str, object]:
    """Summarize subagent findings → state.summary (raw findings left for verify_citations)."""
    findings = state.get("findings", [])
    if not findings:
        return {"summary": "(no findings to compact)"}

    summary, usage = compact_findings(findings, state.get("lead_model", LEAD_MODEL))  # type: ignore[arg-type]
    return {"summary": summary, "token_usage": [usage] if usage else []}
