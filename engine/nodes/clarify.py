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

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langgraph.types import interrupt
from pydantic import BaseModel

from engine.models import LEAD_MODEL
from engine.state import Clarification, ResearchState


class ClarifyDecision(BaseModel):
    is_ambiguous: bool
    questions: list[str]  # 1–3 questions if ambiguous, empty otherwise
    refined_query: str    # original query if not ambiguous; unchanged if ambiguous


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research assistant. Decide whether the given research query is too "
        "ambiguous to research effectively without clarification.\n\n"
        "A query is ambiguous when it could mean meaningfully different things "
        "(e.g. 'Tell me about Mistral' — is it the AI company, the wind, or something else?) "
        "or when a critical scope parameter is missing.\n\n"
        "If ambiguous: set is_ambiguous=true, write 1–3 short clarifying questions, "
        "and set refined_query to the original query unchanged.\n"
        "If clear enough: set is_ambiguous=false, questions=[], "
        "and refined_query to a slightly cleaned-up version of the original query.",
    ),
    ("human", "Query: {query}"),
])


def clarify(state: ResearchState) -> dict[str, object]:
    """Detect ambiguity and store questions in state (LLM call — runs ONCE, not on resume)."""
    # If questions are already stored, a previous run already decided — skip the LLM.
    if state.get("clarification_questions"):
        return {}

    llm: ChatOpenAI = ChatOpenAI(model=LEAD_MODEL, temperature=0)
    chain = _PROMPT | llm.with_structured_output(ClarifyDecision, method="function_calling")
    decision: ClarifyDecision = chain.invoke({"query": state["query"]})  # type: ignore[assignment]

    if not decision.is_ambiguous:
        return {
            "query": decision.refined_query,
            "clarification_questions": [],
            "clarifications": [],
        }

    # Store questions in state so clarify_wait can read them (and so the LLM
    # doesn't need to be called again if the graph is resumed).
    return {"clarification_questions": decision.questions}


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
