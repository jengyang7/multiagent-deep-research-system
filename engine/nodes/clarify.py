"""HUMAN-IN-THE-LOOP clarify nodes (two-node design):

clarify       — calls the LLM once to detect ambiguity and store questions in state.
                Only runs on the FIRST pass; never re-runs on resume.
clarify_wait  — calls interrupt() with the stored questions.
                On first pass: pauses the graph (GraphInterrupt raised).
                On resume via Command(resume=answers): returns answers immediately,
                without re-calling the LLM.

Splitting these two responsibilities across two nodes is the canonical LangGraph
pattern for human-in-the-loop: the node that calls interrupt() is the one that
gets re-run on resume, so all LLM work is safely isolated in the earlier node.
"""
from __future__ import annotations

from datetime import date

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel

from engine.models import LEAD_MODEL
from engine.state import Clarification, ResearchState


class QuestionWithOptions(BaseModel):
    question: str
    options: list[str]  # 3–4 short chip labels (max ~30 chars each)


class ClarifyDecision(BaseModel):
    is_ambiguous: bool
    questions_with_options: list[QuestionWithOptions]  # 1–3 items if ambiguous, empty otherwise
    refined_query: str  # original query if not ambiguous; unchanged if ambiguous


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research assistant. Before deep research begins, ask 1–3 focused "
        "clarifying questions to tailor the research to what the user actually needs.\n\n"
        "Ask clarifying questions for ALMOST ALL research queries. Most queries benefit "
        "from clarification on scope, time frame, geography, target audience, angle, "
        "depth, or purpose.\n\n"
        "ONLY skip clarification (is_ambiguous=false) when the query is an extremely "
        "specific factual lookup with one definitive answer "
        "(e.g. 'What is the capital of France?', 'What year was Python created?').\n\n"
        "For each question you generate, also provide 3–4 short chip options the user can "
        "tap as quick answers. Options must be concise (≤30 chars), mutually exclusive, "
        "and cover the most likely choices. Always include a variety option like "
        "'All of the above' or 'General overview' where appropriate.\n\n"
        "Example output for 'How is the SWE job market in Singapore?':\n"
        "  question: 'Who is this research for?'\n"
        "  options: ['Job seeker', 'Employer / hiring', 'Investor', 'General curiosity']\n\n"
        "  question: 'What time frame should the report focus on?'\n"
        "  options: ['2025 only', '2023–2025', 'Long-term outlook', 'Historical trend']\n\n"
        "Keep questions concise (one sentence). Do not ask redundant questions.\n\n"
        "Today's date: {current_date}. Use the actual current year when writing time-related "
        "chip options — never hardcode past years.",
    ),
    ("human", "Query: {query}"),
])


def clarify(state: ResearchState) -> dict[str, object]:
    """Detect ambiguity and store questions+options in state (LLM call — runs ONCE, not on resume)."""
    # If questions are already stored, a previous run already decided — skip the LLM.
    if state.get("clarification_questions"):
        return {}

    llm: ChatOpenAI = ChatOpenAI(model=LEAD_MODEL, temperature=0)
    chain = _PROMPT | llm.with_structured_output(ClarifyDecision, method="function_calling")
    decision: ClarifyDecision = chain.invoke(  # type: ignore[assignment]
        {"query": state["query"], "current_date": date.today().strftime("%B %d, %Y")}
    )

    if not decision.is_ambiguous:
        return {
            "query": decision.refined_query,
            "clarification_questions": [],
            "clarification_options": [],
            "clarifications": [],
        }

    # Store both questions and chip options so the API can forward them to the UI.
    return {
        "clarification_questions": [q.question for q in decision.questions_with_options],
        "clarification_options": [q.options for q in decision.questions_with_options],
    }


def clarify_wait(state: ResearchState) -> dict[str, object]:
    """Interrupt for user answers if questions are pending (re-runs safely on resume)."""
    questions = state.get("clarification_questions", [])
    if not questions:
        return {}

    # interrupt() raises GraphInterrupt on first pass (pauses the graph).
    # On resume via Command(resume=answers), it returns answers immediately.
    answers: list[str] = interrupt(questions)

    clarifications: list[Clarification] = [
        Clarification(question=q, answer=a)
        for q, a in zip(questions, answers)
    ]
    answers_text = "; ".join(
        f"{c['question']} → {c['answer']}" for c in clarifications
    )
    refined = f"{state['query']} (clarifications: {answers_text})"

    return {
        "query": refined,
        "clarifications": clarifications,
        "clarification_questions": [],  # clear after use
    }
