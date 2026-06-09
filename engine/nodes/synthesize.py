from __future__ import annotations

from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from engine.models import LEAD_MODEL
from engine.state import ResearchState

_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a research synthesizer. Write a comprehensive, well-structured Markdown "
        "report answering the original research query using only the provided findings.\n\n"
        "Requirements:\n"
        "- Use ## and ### headers to organize sections logically\n"
        "- Use bullet lists (- item) for grouped facts; numbered lists for steps\n"
        "- Cite sources with numbered superscripts: [1], [2], [3] etc. after each claim\n"
        "- Do NOT embed full URLs or hyperlinks inside the body text\n"
        "- Do not introduce any information not present in the provided findings\n"
        "- End with a ## References section formatted exactly as:\n"
        "  [1] [Source Title](url)\n"
        "  [2] [Source Title](url)\n"
        "- If findings are sparse or contradictory, note it explicitly in the report",
    ),
    (
        "human",
        "Research query: {query}\n\nFindings:\n{findings_text}",
    ),
])


def _format_findings(findings: list[dict[str, str]]) -> str:
    if not findings:
        return "(no findings collected)"
    lines = []
    for i, f in enumerate(findings, 1):
        lines.append(
            f"[{i}] Subtask: {f['subtask']}\n"
            f"    Claim: {f['claim']}\n"
            f"    Evidence: {f['evidence_span']}\n"
            f"    Source: {f['citation_url']}"
        )
    return "\n\n".join(lines)


def synthesize(state: ResearchState) -> dict[str, str]:
    """Write a cited Markdown report from compacted summary or raw findings (synthesize node)."""
    # Prefer the compacted summary (layer 2) — fall back to raw findings if compact was skipped
    summary = state.get("summary", "")
    findings_text = summary if summary else _format_findings(state.get("findings", []))  # type: ignore[arg-type]
    llm: ChatOpenAI = ChatOpenAI(model=LEAD_MODEL, temperature=0)
    chain = _PROMPT | llm
    result: BaseMessage = chain.invoke(  # type: ignore[assignment]
        {"query": state["query"], "findings_text": findings_text}
    )
    return {"report": str(result.content)}
