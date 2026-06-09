import operator
from typing import Annotated

from typing_extensions import TypedDict


class SubtaskFinding(TypedDict):
    subtask: str
    claim: str
    evidence_span: str
    citation_url: str


# WORKING MEMORY (layer 1 of the memory stack):
# Typed LangGraph state — the in-run scratchpad passed between all nodes.
# Ephemeral; lives only for the duration of a single graph execution.
class ResearchState(TypedDict):
    run_id: str
    query: str
    # Sub-questions produced by the plan node
    subtasks: list[str]
    # operator.add reducer accumulates findings from all parallel subagents
    findings: Annotated[list[SubtaskFinding], operator.add]
    # Populated by the compact node (context compaction, layer 2) — empty in Phase 1
    summary: str
    # Final cited Markdown report written by the synthesize node
    report: str


class SubagentInput(TypedDict):
    """Payload sent to each subagent via Send fan-out. Not the full graph state."""
    question: str
