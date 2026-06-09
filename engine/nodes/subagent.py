from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from engine.extraction import FindingList
from engine.models import SUBAGENT_MODEL
from engine.state import SubagentInput, SubtaskFinding
from engine.tools.fetch import fetch
from engine.tools.search import search

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research extraction agent. Given a sub-question and web content, "
        "extract every relevant finding. Each finding requires:\n"
        "- claim: a clear, factual statement directly supported by the content\n"
        "- evidence_span: the exact quote or passage from the content that supports the claim\n"
        "- citation_url: the URL of the source\n\n"
        "Only include findings directly supported by the provided content. "
        "If no relevant findings exist, return an empty list.",
    ),
    (
        "human",
        "Sub-question: {question}\n\nSource URL: {url}\n\nContent:\n{content}",
    ),
])


def subagent(state: SubagentInput) -> dict[str, list[SubtaskFinding]]:
    """search → fetch → extract validated Findings for one sub-question.

    Each Send fan-out invocation handles exactly one subtask. Results are merged
    back into ResearchState.findings via the operator.add reducer.
    """
    question = state["question"]
    llm: ChatOpenAI = ChatOpenAI(model=SUBAGENT_MODEL, temperature=0)
    chain = _PROMPT | llm.with_structured_output(FindingList, method="function_calling")

    results = search(question, max_results=4)
    findings: list[SubtaskFinding] = []

    for result in results:
        url: str = result.get("url", "")
        if not url:
            continue

        # Prefer the fetched body; fall back to Tavily's snippet if fetch fails
        content: str = fetch(url) or result.get("content", "")
        if not content:
            continue

        try:
            extracted: FindingList = chain.invoke(  # type: ignore[assignment]
                {"question": question, "url": url, "content": content[:6_000]}
            )
            for f in extracted.findings:
                findings.append(
                    SubtaskFinding(
                        subtask=question,
                        claim=f.claim,
                        evidence_span=f.evidence_span,
                        citation_url=str(f.citation_url),
                    )
                )
        except Exception:
            # Silently drop unextractable sources; don't let one bad page fail the subtask
            continue

    return {"findings": findings}
