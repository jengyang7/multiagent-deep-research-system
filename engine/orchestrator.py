from __future__ import annotations

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Send

from engine.nodes.clarify import clarify, clarify_wait
from engine.nodes.compact import compact
from engine.nodes.debate import (
    debate_advocate,
    debate_skeptic,
    judge_debate,
    plan_gap_research,
)
from engine.nodes.plan import plan
from engine.nodes.subagent import subagent
from engine.nodes.synthesize import synthesize
from engine.nodes.verify_citations import verify_citations
from engine.state import ResearchState, SubagentInput

DEFAULT_DEBATE_ROUNDS = 2


def _fan_out(state: ResearchState) -> list[Send]:
    """Conditional edge: plan → one Send per subtask (parallel fan-out)."""
    return [Send("subagent", SubagentInput(question=q)) for q in state["subtasks"]]


def _route_after_compact(state: ResearchState) -> str:
    """Conditional edge: enter the debate loop only when debate mode is on."""
    return "debate_advocate" if state.get("debate_mode") else "synthesize"


def _route_after_skeptic(state: ResearchState) -> str:
    """Conditional edge: loop back to the advocate until the configured rounds are done,
    then have the neutral lead judge the finished debate."""
    rounds_done = len(state.get("debate_turns", [])) // 2
    if rounds_done < state.get("debate_rounds", DEFAULT_DEBATE_ROUNDS):
        return "debate_advocate"
    return "judge_debate"


def _fan_out_gaps(state: ResearchState) -> list[Send] | str:
    """Conditional edge: one Send per gap question, or straight to synthesize
    when the debate surfaced no material evidence gaps."""
    gaps = state.get("gap_subtasks", [])
    if not gaps:
        return "synthesize"
    return [Send("gap_subagent", SubagentInput(question=q)) for q in gaps]


def build_graph(checkpointer: BaseCheckpointSaver | None = None) -> CompiledStateGraph:  # type: ignore[type-arg]
    """Build and compile the research graph.

    Graph flow:
        START → clarify → clarify_wait (interrupt if ambiguous) → plan
              → N parallel subagents → compact (layer 2)
              → [debate mode only: debate_advocate ⇄ debate_skeptic × N rounds
                 → judge_debate → plan_gap_research
                 → M parallel gap_subagents → recompact]
              → synthesize → verify_citations → END

    Debate mode (off by default): two cross-provider agents argue over the
    compacted findings before synthesis. Each turn is its own node execution,
    so turns stream individually and checkpoint per turn. After the final
    round, judge_debate (neutral lead model) declares a winner for the UI
    verdict card. The debate then drives a second research round:
    plan_gap_research distills unresolved skeptic objections into follow-up
    questions, gap_subagents (same subagent fn, separate node name so the
    fan-in edge targets recompact, not compact) research them, and recompact
    folds the new findings into state.summary.

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
    builder.add_node("debate_advocate", debate_advocate)  # type: ignore[arg-type]
    builder.add_node("debate_skeptic", debate_skeptic)    # type: ignore[arg-type]
    builder.add_node("judge_debate", judge_debate)        # type: ignore[arg-type]
    builder.add_node("plan_gap_research", plan_gap_research)  # type: ignore[arg-type]
    builder.add_node("gap_subagent", subagent)      # type: ignore[arg-type]
    builder.add_node("recompact", compact)          # type: ignore[arg-type]
    builder.add_node("synthesize", synthesize)      # type: ignore[arg-type]
    builder.add_node("verify_citations", verify_citations)  # type: ignore[arg-type]

    # Graph flow
    builder.add_edge(START, "clarify")
    builder.add_edge("clarify", "clarify_wait")
    builder.add_edge("clarify_wait", "plan")
    builder.add_conditional_edges("plan", _fan_out, ["subagent"])  # type: ignore[arg-type]
    builder.add_edge("subagent", "compact")
    builder.add_conditional_edges(
        "compact", _route_after_compact, ["debate_advocate", "synthesize"]
    )
    builder.add_edge("debate_advocate", "debate_skeptic")
    builder.add_conditional_edges(
        "debate_skeptic", _route_after_skeptic, ["debate_advocate", "judge_debate"]
    )
    builder.add_edge("judge_debate", "plan_gap_research")
    builder.add_conditional_edges(
        "plan_gap_research", _fan_out_gaps, ["gap_subagent", "synthesize"]
    )
    builder.add_edge("gap_subagent", "recompact")
    builder.add_edge("recompact", "synthesize")
    builder.add_edge("synthesize", "verify_citations")
    builder.add_edge("verify_citations", END)

    return builder.compile(checkpointer=checkpointer)  # type: ignore[return-value]


# Module-level graph without checkpointer — for tests and one-shot use
graph: CompiledStateGraph = build_graph()  # type: ignore[type-arg]
