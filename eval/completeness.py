"""LLM-as-judge completeness check (Phase 4 eval harness).

Two-step "must-cover checklist":
  1. Given only the query, ask the judge for the distinct subtopics/aspects a
     thorough report should cover.
  2. Given the report body and that checklist, ask the judge which subtopics
     are actually addressed.

`recall_score = covered / total` (1.0 if the judge returned zero subtopics).
"""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from engine.models import LEAD_MODEL
from engine.state import TokenUsage
from engine.usage import usage_from_message
from eval.report_parsing import split_body_and_references
from eval.schema import CompletenessResult, SubtopicCoverage

_MAX_SUBTOPICS = 8


class _SubtopicList(BaseModel):
    """Structured-output schema for the subtopic-generation judge."""

    subtopics: list[str] = Field(
        description=(
            f"Up to {_MAX_SUBTOPICS} distinct subtopics or aspects that a thorough "
            "research report answering this query should cover. Each entry should "
            "be a short phrase (a few words)."
        )
    )


class _CoverageList(BaseModel):
    """Structured-output schema for the coverage-scoring judge."""

    coverage: list[SubtopicCoverage] = Field(
        description="One entry per given subtopic, in the same order, with `covered` "
        "set to true only if the report substantively addresses that subtopic."
    )


_SUBTOPICS_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are designing a grading rubric for a research report. Given a "
        "user's query, list the distinct subtopics or aspects a thorough "
        f"report answering it should cover, up to {_MAX_SUBTOPICS} items. "
        "Keep each item short (a few words). Order from most to least important.",
    ),
    ("human", "Query: {query}"),
])

_COVERAGE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are grading a research report against a checklist of expected "
        "subtopics. For each subtopic, determine whether the report "
        "substantively addresses it (covered=true) or not (covered=false), "
        "with a brief note explaining your judgment. Return one entry per "
        "subtopic, in the same order given.",
    ),
    (
        "human",
        "Expected subtopics:\n{subtopics}\n\nReport:\n{report_body}",
    ),
])


async def generate_expected_subtopics(
    query: str, lead_model: str = LEAD_MODEL
) -> tuple[list[str], TokenUsage | None]:
    """Ask the judge what subtopics a thorough report on `query` should cover."""
    judge_llm = ChatOpenAI(model=lead_model, temperature=0)
    chain = _SUBTOPICS_PROMPT | judge_llm.with_structured_output(
        _SubtopicList, method="function_calling", include_raw=True
    )
    raw = await chain.ainvoke({"query": query})
    result: _SubtopicList = raw["parsed"]
    usage = usage_from_message(raw["raw"], "completeness_subtopics", lead_model)
    return result.subtopics[:_MAX_SUBTOPICS], usage


async def score_completeness(
    report: str, subtopics: list[str], lead_model: str = LEAD_MODEL
) -> tuple[CompletenessResult, TokenUsage | None]:
    """Score which `subtopics` are covered by `report`."""
    if not subtopics:
        return CompletenessResult(subtopics=[], recall_score=1.0), None

    body, _ = split_body_and_references(report)

    judge_llm = ChatOpenAI(model=lead_model, temperature=0)
    chain = _COVERAGE_PROMPT | judge_llm.with_structured_output(
        _CoverageList, method="function_calling", include_raw=True
    )
    raw = await chain.ainvoke({
        "subtopics": "\n".join(f"- {s}" for s in subtopics),
        "report_body": body,
    })
    result: _CoverageList = raw["parsed"]
    usage = usage_from_message(raw["raw"], "completeness_coverage", lead_model)

    covered = sum(1 for c in result.coverage if c.covered)
    total = len(result.coverage)
    recall_score = covered / total if total else 1.0
    return CompletenessResult(subtopics=result.coverage, recall_score=recall_score), usage


async def run_completeness_check(
    query: str, report: str, lead_model: str = LEAD_MODEL
) -> tuple[CompletenessResult, list[TokenUsage]]:
    """Generate the expected-subtopics checklist for `query`, then score `report` against it."""
    subtopics, subtopics_usage = await generate_expected_subtopics(query, lead_model)
    result, coverage_usage = await score_completeness(report, subtopics, lead_model)

    usages = [u for u in (subtopics_usage, coverage_usage) if u is not None]
    return result, usages
