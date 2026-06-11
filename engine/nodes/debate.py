"""Adversarial debate nodes (debate mode).

Two agents from different AI companies argue over the compacted findings
before synthesis: the advocate builds the strongest well-supported answer,
the skeptic attacks evidence quality and overreach. Each node execution is
one turn, so the loop edges in the orchestrator stream every turn to the UI
via the existing stream_mode="updates" SSE pipeline, checkpoint per turn,
and accumulate token usage through the operator.add reducer.

Cross-provider debaters are the point: different pretraining/RLHF lineages
have uncorrelated blind spots, so the skeptic catches gaps a same-model
skeptic would agree with. Models fall back to lead_model when a provider
key is missing.
"""
from __future__ import annotations

from typing import Literal

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from engine.models import LEAD_MODEL, make_chat_model, structured_output_kwargs
from engine.state import DebateTurn, DebateVerdict, ResearchState
from engine.usage import usage_from_message

MAX_GAP_QUESTIONS = 3

_ADVOCATE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the ADVOCATE in a structured research debate. Your job is to "
        "argue the strongest, best-supported answer to the research query using "
        "ONLY the claims in the research summary below — never invent facts.\n"
        "- Build a clear position: what does the evidence, taken together, "
        "support most strongly?\n"
        "- Lean on specific claims and figures, and name the source URL when a "
        "point rests on it.\n"
        "- If a skeptic has already spoken, rebut their strongest objections "
        "directly: concede points the evidence cannot answer, and defend points "
        "it can.\n"
        "- Be concise and substantive: 2-4 tight paragraphs, no preamble.",
    ),
    (
        "human",
        "Research query: {query}\n\nResearch summary:\n{summary}\n\n"
        "Debate so far:\n{transcript}\n\nYour turn, Advocate (round {round}).",
    ),
])

_SKEPTIC_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the SKEPTIC in a structured research debate. Your job is to "
        "stress-test the advocate's last argument using ONLY the research "
        "summary below — never invent counter-facts.\n"
        "- Attack evidence quality: single-source claims, missing data, vague "
        "figures, sources that don't actually support the weight put on them.\n"
        "- Surface gaps, contradictions between findings, and overreach — "
        "places where the advocate's conclusion goes beyond what the summary "
        "states.\n"
        "- Point out what a careful reader would still need to know before "
        "accepting the position.\n"
        "- Concede genuinely strong points; a skeptic who disputes everything "
        "is useless.\n"
        "- Be concise and substantive: 2-4 tight paragraphs, no preamble.",
    ),
    (
        "human",
        "Research query: {query}\n\nResearch summary:\n{summary}\n\n"
        "Debate so far:\n{transcript}\n\nYour turn, Skeptic (round {round}).",
    ),
])


def format_transcript(turns: list[DebateTurn]) -> str:
    """Render prior turns for the debaters' prompts and the synthesizer."""
    if not turns:
        return "(the debate is just beginning)"
    return "\n\n".join(
        f"[Round {t['round']} — {t['agent'].capitalize()}]\n{t['content']}" for t in turns
    )


def _run_turn(
    state: ResearchState, agent: str, model: str, prompt: ChatPromptTemplate
) -> dict[str, object]:
    turns = state.get("debate_turns", [])
    # Advocate speaks at even turn counts, skeptic at odd — same round formula for both
    round_no = len(turns) // 2 + 1
    # Slight temperature for argumentative diversity (every other node runs at 0)
    llm = make_chat_model(model, temperature=0.4)
    chain = prompt | llm
    result: BaseMessage = chain.invoke({
        "query": state["query"],
        "summary": state.get("summary", ""),
        "transcript": format_transcript(turns),
        "round": round_no,
    })
    usage = usage_from_message(result, f"debate_{agent}", model)
    # .text, not str(.content): Gemini returns a list of content blocks, which
    # would otherwise render as a raw "[{'type': 'text', ...}]" python literal
    turn = DebateTurn(agent=agent, model=model, round=round_no, content=result.text)
    return {"debate_turns": [turn], "token_usage": [usage] if usage else []}


def debate_advocate(state: ResearchState) -> dict[str, object]:
    """Argue the strongest evidence-backed position (debate mode, turn node)."""
    model = state.get("advocate_model") or state.get("lead_model", LEAD_MODEL)
    return _run_turn(state, "advocate", model, _ADVOCATE_PROMPT)


def debate_skeptic(state: ResearchState) -> dict[str, object]:
    """Challenge evidence quality, gaps, and overreach (debate mode, turn node)."""
    model = state.get("skeptic_model") or state.get("lead_model", LEAD_MODEL)
    return _run_turn(state, "skeptic", model, _SKEPTIC_PROMPT)


# ---------------------------------------------------------------------------
# Debate judgment: after the final round, the (neutral) lead model weighs both
# sides and declares a winner. The verdict is purely informational — it feeds
# the UI verdict card and history; gap planning and synthesis are unaffected.
# ---------------------------------------------------------------------------

class DebateJudgment(BaseModel):
    reasoning: str  # 2-4 sentences: which arguments held up and which collapsed
    winner: Literal["advocate", "skeptic", "draw"]


_JUDGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are the neutral JUDGE of a structured research debate between an "
        "advocate (argues the best-supported answer) and a skeptic (attacks "
        "evidence quality and overreach). Decide who argued better — judge the "
        "ARGUMENTS, not which position you personally favor.\n"
        "- The advocate wins if their position survived the skeptic's strongest "
        "objections with evidence from the summary.\n"
        "- The skeptic wins if they exposed material gaps, contradictions, or "
        "overreach the advocate could not answer.\n"
        "- Call it a draw only when the rounds are genuinely balanced.\n"
        "- In 'reasoning', give 2-4 plain-language sentences citing the decisive "
        "moments of the debate.",
    ),
    (
        "human",
        "Research query: {query}\n\nResearch summary:\n{summary}\n\n"
        "Debate transcript:\n{transcript}\n\nYour verdict, Judge.",
    ),
])


def judge_debate(state: ResearchState) -> dict[str, object]:
    """Declare the debate winner from a neutral lead-model judgment (debate mode)."""
    model = state.get("lead_model", LEAD_MODEL)
    llm = make_chat_model(model, temperature=0)
    chain = _JUDGE_PROMPT | llm.with_structured_output(
        DebateJudgment, include_raw=True, **structured_output_kwargs(model)
    )
    raw = chain.invoke({
        "query": state["query"],
        "summary": state.get("summary", ""),
        "transcript": format_transcript(state.get("debate_turns", [])),
    })
    assert isinstance(raw, dict)  # include_raw=True returns {"raw", "parsed", "parsing_error"}
    result: DebateJudgment = raw["parsed"]
    usage = usage_from_message(raw["raw"], "judge_debate", model)
    verdict = DebateVerdict(winner=result.winner, reasoning=result.reasoning, model=model)
    return {"debate_verdict": verdict, "token_usage": [usage] if usage else []}


# ---------------------------------------------------------------------------
# Debate-driven gap research: after the final round, the (neutral) lead model
# distills the skeptic's unresolved objections into concrete follow-up search
# questions. A second subagent fan-out researches them before synthesis, so
# the report answers the debate's open questions with evidence instead of
# leaving them as caveats.
# ---------------------------------------------------------------------------

class GapResearchPlan(BaseModel):
    thinking: str  # Brief reasoning: which objections survived rebuttal and need evidence
    gap_questions: list[str]


_GAP_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a neutral research lead reviewing an adversarial debate over a "
        "research summary. Identify the evidence GAPS that block a confident "
        "answer: objections the skeptic raised that the advocate could not rebut "
        "with the existing findings, and facts both sides agreed were missing.\n"
        f"- In 'gap_questions': write 0–{MAX_GAP_QUESTIONS} follow-up research "
        "questions targeting those gaps. Each must be self-contained, concrete, "
        "and directly answerable via a web search (name the specific data, "
        "comparison, or timeframe needed).\n"
        "- Do NOT re-ask what the summary already answers, and do not restate "
        "debate rhetoric — only genuinely missing evidence qualifies.\n"
        "- If the debate surfaced no material gaps, return an empty list.",
    ),
    (
        "human",
        "Research query: {query}\n\nResearch summary:\n{summary}\n\n"
        "Debate transcript:\n{transcript}",
    ),
])


def plan_gap_research(state: ResearchState) -> dict[str, object]:
    """Distill unresolved debate objections into follow-up search questions (debate mode)."""
    model = state.get("lead_model", LEAD_MODEL)
    llm = make_chat_model(model, temperature=0)
    chain = _GAP_PROMPT | llm.with_structured_output(
        GapResearchPlan, include_raw=True, **structured_output_kwargs(model)
    )
    raw = chain.invoke({
        "query": state["query"],
        "summary": state.get("summary", ""),
        "transcript": format_transcript(state.get("debate_turns", [])),
    })
    assert isinstance(raw, dict)  # include_raw=True returns {"raw", "parsed", "parsing_error"}
    result: GapResearchPlan = raw["parsed"]
    usage = usage_from_message(raw["raw"], "plan_gap_research", model)
    return {
        "gap_subtasks": result.gap_questions[:MAX_GAP_QUESTIONS],
        "token_usage": [usage] if usage else [],
    }
