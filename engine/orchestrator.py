from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from engine.nodes.clarify import clarify, clarify_wait
from engine.nodes.compact import compact
from engine.nodes.plan import plan
from engine.nodes.subagent import subagent
from engine.nodes.synthesize import synthesize
from engine.state import ResearchState, SubagentInput


def _fan_out(state: ResearchState) -> list[Send]:
    """Conditional edge: plan → one Send per subtask (parallel fan-out)."""
    return [Send("subagent", SubagentInput(question=q)) for q in state["subtasks"]]


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build and compile the research graph.

    Graph flow:
        START → clarify → clarify_wait (interrupt if ambiguous) → plan
              → N parallel subagents → compact (layer 2) → synthesize → END

    Two-node clarify design: clarify calls the LLM once; clarify_wait holds the
    interrupt() so the LLM is never re-called on resume.

    Pass a checkpointer (layer 3) to enable resumable runs and human-in-the-loop.
    Omit it for tests / one-shot runs that don't need persistence.
    """
    builder: StateGraph = StateGraph(ResearchState)  # type: ignore[type-arg]

    builder.add_node("clarify", clarify)            # type: ignore[arg-type]
    builder.add_node("clarify_wait", clarify_wait)  # type: ignore[arg-type]
    builder.add_node("plan", plan)                  # type: ignore[arg-type]
    builder.add_node("subagent", subagent)          # type: ignore[arg-type]
    builder.add_node("compact", compact)            # type: ignore[arg-type]
    builder.add_node("synthesize", synthesize)      # type: ignore[arg-type]

    # Graph flow
    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", "clarify_wait")
    builder.add_edge("clarify_wait", "plan")
    builder.add_conditional_edges("plan", _fan_out, ["subagent"])  # type: ignore[arg-type]
    builder.add_edge("subagent", "compact")
    builder.add_edge("compact", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile(checkpointer=checkpointer)  # type: ignore[return-value]


# Module-level graph without checkpointer — for tests and one-shot use
graph: CompiledStateGraph = build_graph()  # type: ignore[type-arg]
