"""LLM-as-judge relevance check (Phase 4 eval harness).

Single judge call: given the original query and the report body, score how
directly the report responds to the query on a 1-5 scale, with reasoning.
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from engine.models import LEAD_MODEL, make_chat_model, structured_output_kwargs
from engine.state import TokenUsage
from engine.usage import usage_from_message
from eval.report_parsing import split_body_and_references
from eval.schema import RelevanceResult


class _RelevanceVerdict(BaseModel):
    """Structured-output schema for the relevance judge."""

    score: int = Field(
        ge=1, le=5,
        description=(
            "1 = report is off-topic or does not address the query; "
            "5 = report is squarely focused on and directly responsive to the query."
        ),
    )
    reasoning: str = Field(description="One or two sentence justification")


_RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are grading a research report for relevance. Score 1-5 how "
        "directly and fully the report addresses the user's query — penalize "
        "reports that drift into tangential topics, ignore the query's intent, "
        "or answer a different question than the one asked.",
    ),
    ("human", "Query: {query}\n\nReport:\n{report_body}"),
])


async def run_relevance_check(
    query: str, report: str, lead_model: str = LEAD_MODEL
) -> tuple[RelevanceResult, TokenUsage | None]:
    """Score how relevant `report` is to `query`."""
    body, _ = split_body_and_references(report)

    judge_llm = make_chat_model(lead_model, temperature=0)
    chain = _RELEVANCE_PROMPT | judge_llm.with_structured_output(
        _RelevanceVerdict, **structured_output_kwargs(lead_model), include_raw=True
    )
    raw = await chain.ainvoke({"query": query, "report_body": body})
    assert isinstance(raw, dict)  # include_raw=True returns {"raw", "parsed", "parsing_error"}
    verdict: _RelevanceVerdict = raw["parsed"]
    usage = usage_from_message(raw["raw"], "relevance", lead_model)

    return RelevanceResult(score=verdict.score, reasoning=verdict.reasoning), usage
