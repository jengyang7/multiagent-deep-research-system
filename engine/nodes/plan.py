from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from engine.models import LEAD_MODEL
from engine.state import ResearchState


class ResearchPlan(BaseModel):
    thinking: str  # Supervisor's brief reasoning about how to approach the query
    subtasks: list[str]


_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research planner. Given a research query:\n"
        "1. In 'thinking': write 2-3 sentences explaining your approach — what the user "
        "   wants, what key angles to cover, and why you chose these sub-questions.\n"
        "2. In 'subtasks': decompose into 3–6 independent, specific sub-questions that "
        "   together fully cover the topic. Each must be self-contained and directly "
        "   answerable via a web search. Do not overlap. Prefer concrete, searchable phrasing.",
    ),
    ("human", "Research query: {query}"),
])


def plan(state: ResearchState) -> dict:  # type: ignore[type-arg]
    """Decompose the research query into parallel sub-questions (plan node)."""
    llm: ChatOpenAI = ChatOpenAI(model=LEAD_MODEL, temperature=0)
    chain = _PROMPT | llm.with_structured_output(ResearchPlan, method="function_calling")
    result: ResearchPlan = chain.invoke({"query": state["query"]})  # type: ignore[assignment]
    return {"subtasks": result.subtasks, "supervisor_thinking": result.thinking}
