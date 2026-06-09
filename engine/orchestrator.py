from __future__ import annotations

from langgraph.constants import Send
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from engine.nodes.plan import plan
from engine.nodes.subagent import subagent
from engine.nodes.synthesize import synthesize
from engine.state import ResearchState, SubagentInput


def _fan_out(state: ResearchState) -> list[Send]:
    """Conditional edge: plan → one Send per subtask (parallel fan-out)."""
    return [Send("subagent", SubagentInput(question=q)) for q in state["subtasks"]]


def _build_graph() -> CompiledStateGraph:  # type: ignore[type-arg]
    builder: StateGraph = StateGraph(ResearchState)  # type: ignore[type-arg]

    builder.add_node("plan", plan)       # type: ignore[arg-type]
    builder.add_node("subagent", subagent)  # type: ignore[arg-type]
    builder.add_node("synthesize", synthesize)  # type: ignore[arg-type]

    # Graph flow: start → plan → N parallel subagents → synthesize → end
    builder.add_edge(START, "plan")
    builder.add_conditional_edges("plan", _fan_out, ["subagent"])  # type: ignore[arg-type]
    builder.add_edge("subagent", "synthesize")
    builder.add_edge("synthesize", END)

    return builder.compile()  # type: ignore[return-value]


# Module-level compiled graph — import and invoke directly
graph: CompiledStateGraph = _build_graph()  # type: ignore[type-arg]
