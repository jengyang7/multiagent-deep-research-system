"""Load a completed research run for evaluation.

The `subtasks`/`sources`/`findings`/`reports` tables in db/models.py are
defined but never written to — the actual run output (report + findings)
lives only in the LangGraph Postgres checkpointer state for
thread_id == run_id, accessed the same way api/main.py does.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from db.models import ResearchRun
from engine.memory.checkpointer import get_checkpointer
from engine.orchestrator import build_graph
from engine.state import SubtaskFinding


@dataclass
class EvalRunData:
    run_id: str
    query: str
    status: str
    report: str
    findings: list[SubtaskFinding]


def _async_engine_from_url(raw: str) -> AsyncEngine:
    """Build an asyncpg-compatible engine, stripping params asyncpg doesn't accept.

    Mirrors api/main.py:_async_engine_from_url.
    """
    parsed = urlparse(raw)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    ssl_val = params.pop("ssl", None) or ("require" if params.pop("sslmode", None) else None)
    for unsupported in ("channel_binding", "options"):
        params.pop(unsupported, None)
    clean_url = urlunparse(parsed._replace(query=urlencode(params)))
    kwargs = {"connect_args": {"ssl": ssl_val}} if ssl_val else {}
    return create_async_engine(clean_url, **kwargs)


def _dedupe_findings(findings: list[SubtaskFinding]) -> list[SubtaskFinding]:
    seen: set[tuple[str, str, str, str]] = set()
    deduped: list[SubtaskFinding] = []
    for f in findings:
        key = (f["subtask"], f["claim"], f["evidence_span"], f["citation_url"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(f)
    return deduped


async def load_run(run_id: str, require_done: bool = True) -> EvalRunData:
    """Load a completed run's report + findings for evaluation.

    - `research_runs` row (Postgres) supplies query/status.
    - The checkpointer's final state supplies `report`.
    - `findings` is empty in the final state (cleared by the verify_citations
      node, the last to use them), so we walk checkpoint history for the most
      recent snapshot with non-empty `findings` (i.e. right after the subagent
      fan-out, before compact/verify_citations).

    Raises:
        ValueError: run not found, or (if require_done) not yet finished.
    """
    engine = _async_engine_from_url(os.environ["DATABASE_URL"])
    session_factory = async_sessionmaker(engine, expire_on_commit=False)

    try:
        async with session_factory() as session:
            result = await session.execute(select(ResearchRun).where(ResearchRun.id == run_id))
            run = result.scalar_one_or_none()
    finally:
        await engine.dispose()

    if run is None:
        raise ValueError(f"Run {run_id} not found")
    if require_done and run.status != "done":
        raise ValueError(f"Run {run_id} has not finished (status={run.status})")

    config = {"configurable": {"thread_id": run_id}}
    async with get_checkpointer() as checkpointer:
        graph = build_graph(checkpointer)

        snapshot = await graph.aget_state(config)  # type: ignore[arg-type]
        report: str = snapshot.values.get("report", "")

        findings: list[SubtaskFinding] = list(snapshot.values.get("findings", []))
        if not findings:
            async for hist in graph.aget_state_history(config):  # type: ignore[arg-type]
                hist_findings = hist.values.get("findings", [])
                if hist_findings:
                    findings = list(hist_findings)
                    break

    return EvalRunData(
        run_id=run_id,
        query=run.query,
        status=run.status,
        report=report,
        findings=_dedupe_findings(findings),
    )
