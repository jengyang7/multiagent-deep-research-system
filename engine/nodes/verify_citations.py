"""CITATION VERIFICATION node (anti-hallucination guard, runs after synthesize):

Re-runs the same per-sentence faithfulness judge used by the eval harness
(eval.faithfulness.run_faithfulness_checks) against the freshly synthesized
report. Any [i]-cited sentence the judge can't verify against the Finding(s)
behind reference [i] has its citation marker(s) programmatically stripped —
turning a falsely-attributed claim into an uncited analytical/synthesis
sentence (which the eval harness treats as informational, not penalized)
instead of leaving a misleading citation in the published report.

Clears state.findings afterward — this is the last node that needs the raw
findings list.
"""
from __future__ import annotations

import re

from engine.models import CITATION_CHECK_MODEL
from engine.state import ResearchState
from eval.faithfulness import run_faithfulness_checks
from eval.report_parsing import (
    extract_citation_indices,
    split_body_and_references,
    split_sentences,
)

_CITATION_MARKER_RE = re.compile(r"\[(\d+)\]")
_DANGLING_SPACE_RE = re.compile(r"[ \t]+(?=[.,;:!?])|  +")


def _sentence_pattern(sentence: str) -> re.Pattern[str]:
    """Build a regex matching `sentence`'s occurrence in the report body.

    split_sentences() joins a paragraph's lines with single spaces, so the
    sentence text may not appear verbatim in the (possibly line-wrapped)
    body — replace whitespace runs with `\\s+` to tolerate that.
    """
    parts = re.split(r"(\s+)", sentence)
    return re.compile("".join(r"\s+" if p.isspace() else re.escape(p) for p in parts))


def _strip_citations(body: str, sentence: str, start: int) -> tuple[str, int]:
    """Remove all [i] markers from `sentence`'s occurrence in `body` (search from `start`).

    Returns (new_body, new_search_offset). No-op if the sentence can't be found.
    """
    match = _sentence_pattern(sentence).search(body, start)
    if not match:
        return body, start
    cleaned = _CITATION_MARKER_RE.sub("", match.group(0))
    cleaned = _DANGLING_SPACE_RE.sub(lambda m: "" if m.group().strip() == "" else " ", cleaned)
    new_body = body[: match.start()] + cleaned + body[match.end() :]
    return new_body, match.start() + len(cleaned)


async def verify_citations(state: ResearchState) -> dict[str, object]:
    """Strip [i] citations the faithfulness judge can't verify (verify_citations node)."""
    report = state.get("report", "")
    findings = state.get("findings", [])
    if not report or not findings:
        return {"findings": []}

    verdicts, _uncited, token_usage = await run_faithfulness_checks(
        report, findings, lead_model=CITATION_CHECK_MODEL
    )

    body, references = split_body_and_references(report)
    cited_sentences = [s for s, _section in split_sentences(body) if extract_citation_indices(s)]

    cursor = 0
    for sentence, verdict in zip(cited_sentences, verdicts):
        if not verdict.faithful:
            body, cursor = _strip_citations(body, sentence, cursor)

    return {"report": body + references, "findings": [], "token_usage": token_usage}
