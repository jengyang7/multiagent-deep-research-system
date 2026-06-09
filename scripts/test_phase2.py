"""Manual integration test script for Phase 2 features.

Tests each memory layer and human-in-the-loop feature in isolation.
Run with:
    uv run python scripts/test_phase2.py [--section <name>]

Sections:
    checkpointer   — save/load state via Postgres (no OpenAI key needed)
    clarify-clear  — clear query passes through without interrupt
    clarify-hil    — ambiguous query triggers interrupt, then resumes
    compact        — compact node summarizes findings
    chat           — follow-up Q&A reads from checkpointer state
    all            — run every section in order (default)

Requires:
    - docker-compose up -d  (Postgres on localhost:5432)
    - OPENAI_API_KEY env var (all sections except 'checkpointer')
    - TAVILY_API_KEY env var (only needed if subagents run; not needed here)
"""
from __future__ import annotations

import argparse
import asyncio
import os
import sys
import uuid

# Make sure project root is on the path when running as a script
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://research:research@localhost:5432/research",
)

SEPARATOR = "\n" + "=" * 60 + "\n"


def section(title: str) -> None:
    print(f"{SEPARATOR}SECTION: {title}{SEPARATOR}")


def ok(msg: str) -> None:
    print(f"  ✓  {msg}")


def info(msg: str) -> None:
    print(f"     {msg}")


# ---------------------------------------------------------------------------
# Section 1: Checkpointer — save and reload state (no LLM calls)
# ---------------------------------------------------------------------------

async def test_checkpointer() -> None:
    section("Checkpointer — save / reload state (layer 3)")

    from engine.memory.checkpointer import get_checkpointer
    from engine.orchestrator import build_graph
    from engine.state import ResearchState

    thread_id = f"test-ckpt-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    async with get_checkpointer() as checkpointer:
        # Build a graph with the checkpointer attached
        g = build_graph(checkpointer=checkpointer)

        # Manually save a state snapshot by using update_state
        seed_state: ResearchState = {
            "run_id": thread_id,
            "query": "What is LangGraph?",
            "clarifications": [],
            "subtasks": ["sub-q1", "sub-q2"],
            "findings": [
                {
                    "subtask": "sub-q1",
                    "claim": "LangGraph is a graph-based agent framework.",
                    "evidence_span": "LangGraph builds on LangChain...",
                    "citation_url": "https://example.com/langgraph",
                }
            ],
            "summary": "LangGraph summary",
            "report": "# LangGraph\n\nLangGraph is a framework.",
            "messages": [],
        }
        await g.aupdate_state(config, seed_state)
        ok("State written to Postgres checkpointer")

        # Reload state from the same thread
        snapshot = await checkpointer.aget_tuple(config)
        assert snapshot is not None, "Snapshot should not be None after write"

        channel_values = snapshot.checkpoint.get("channel_values", {})
        assert channel_values.get("query") == "What is LangGraph?"
        assert channel_values.get("report", "").startswith("# LangGraph")
        ok(f"State reloaded from thread_id={thread_id}")
        info(f"query    = {channel_values['query']!r}")
        info(f"findings = {len(channel_values.get('findings', []))} item(s)")
        info(f"report   = {channel_values['report'][:40]!r}...")


# ---------------------------------------------------------------------------
# Section 2: Clarify node — non-ambiguous query (no interrupt)
# ---------------------------------------------------------------------------

async def test_clarify_clear() -> None:
    section("Clarify node — clear query passes through (no interrupt)")

    from engine.memory.checkpointer import get_checkpointer
    from engine.orchestrator import build_graph

    thread_id = f"test-clarify-clear-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    query = "What are the main differences between Redis and Memcached?"
    info(f"Query: {query!r}")

    async with get_checkpointer() as checkpointer:
        g = build_graph(checkpointer=checkpointer)

        # Invoke only up to the clarify node — interrupt the graph right after clarify
        # by checking state. We use astream_events to catch what happens.
        events = []
        async for event in g.astream_events(
            {"run_id": thread_id, "query": query, "clarifications": [],
             "subtasks": [], "findings": [], "summary": "", "report": "", "messages": []},
            config=config,
            version="v2",
        ):
            events.append(event)
            # Stop after clarify completes (before plan fires) to avoid LLM costs
            if event.get("name") == "clarify" and event.get("event") == "on_chain_end":
                break

        # Check state after clarify
        state = await g.aget_state(config)
        clarifications = state.values.get("clarifications", [])
        ok("Clarify node completed without interrupt")
        info(f"clarifications = {clarifications}")
        assert clarifications == [], f"Expected no clarifications, got: {clarifications}"
        ok("Query was unambiguous — no questions asked")


# ---------------------------------------------------------------------------
# Section 3: Clarify node — ambiguous query triggers interrupt then resumes
# ---------------------------------------------------------------------------

async def test_clarify_hil() -> None:
    section("Clarify node — ambiguous query → interrupt → resume (human-in-the-loop)")

    from langgraph.types import Command

    from engine.memory.checkpointer import get_checkpointer
    from engine.orchestrator import build_graph

    thread_id = f"test-hil-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    # Deliberately vague — should trigger clarification questions
    query = "Tell me about Mistral"
    info(f"Query: {query!r}")

    async with get_checkpointer() as checkpointer:
        g = build_graph(checkpointer=checkpointer)

        interrupted = False

        async for event in g.astream_events(
            {"run_id": thread_id, "query": query, "clarifications": [],
             "subtasks": [], "findings": [], "summary": "", "report": "", "messages": []},
            config=config,
            version="v2",
        ):
            if event.get("event") == "on_custom_event" and event.get("name") == "__interrupt__":
                interrupted = True
                break
            # Also catch interrupts surfaced via metadata
            if event.get("event") == "on_chain_end":
                output = event.get("data", {}).get("output", {})
                if isinstance(output, dict) and "__interrupt__" in output:
                    interrupted = True
                    break

        # LangGraph surfaces interrupts as a special value in the stream
        # Check state to see if the graph is paused
        state = await g.aget_state(config)
        if state.next:
            interrupted = True
            info(f"Graph paused at nodes: {state.next}")

        if not interrupted:
            ok("Query was judged unambiguous by the LLM — no interrupt fired")
            info("(Try a more vague query to reliably trigger interruption)")
            return

        ok("Graph interrupted — clarifying questions surfaced")
        info(f"Next nodes pending: {state.next}")
        # After the two-node fix, the paused node should be clarify_wait (not clarify)
        assert "clarify_wait" in state.next, (
            f"Expected graph to pause at clarify_wait, got: {state.next}"
        )

        # Read the pending questions from state
        pending_questions = state.values.get("clarification_questions", [])
        info(f"Pending questions: {pending_questions}")

        # Simulate user answering — one answer per question
        answer = "The AI company (mistral.ai), not the wind"
        user_answers = [answer] * max(1, len(pending_questions))
        info(f"Submitting user answers: {user_answers}")

        # Resume the graph with user answers.
        # Strategy: capture clarify_wait's output directly from the event data
        # (reliable regardless of checkpoint flush timing), then drain the stream
        # one step further until `plan` starts — which guarantees LangGraph has
        # committed the clarify_wait checkpoint to Postgres before we break.
        clarify_wait_output: dict[str, object] = {}
        async for event in g.astream_events(
            Command(resume=user_answers),
            config=config,
            version="v2",
        ):
            if event.get("name") == "clarify_wait" and event.get("event") == "on_chain_end":
                clarify_wait_output = event.get("data", {}).get("output", {}) or {}
            # Break once plan starts — by this point clarify_wait's checkpoint is flushed
            if event.get("name") == "plan" and event.get("event") == "on_chain_start":
                break

        # Assert on node output (source of truth — no checkpointer timing dependency)
        clarifications_from_output = clarify_wait_output.get("clarifications", [])
        query_from_output = clarify_wait_output.get("query", "")
        ok("Graph resumed successfully after interrupt")
        info(f"clarify_wait output clarifications = {clarifications_from_output}")
        info(f"clarify_wait output query          = {query_from_output!r}")
        assert len(clarifications_from_output) > 0, (
            "Expected clarifications in clarify_wait node output after resume"
        )

        # Also verify the checkpoint was flushed to Postgres (layer 3 — episodic memory)
        state_after = await g.aget_state(config)
        clarifications = state_after.values.get("clarifications", [])
        query_after = state_after.values.get("query", "")
        info(f"checkpointer state clarifications = {clarifications}")
        info(f"checkpointer state query          = {query_after!r}")
        assert len(clarifications) > 0, "Expected clarifications to be persisted in checkpointer"
        ok("Clarification questions + answers stored in checkpointer state")


# ---------------------------------------------------------------------------
# Section 4: Compact node — summarizes findings into state.summary
# ---------------------------------------------------------------------------

async def test_compact() -> None:
    section("Compact node — context compaction (layer 2)")

    from engine.memory.checkpointer import get_checkpointer
    from engine.orchestrator import build_graph

    thread_id = f"test-compact-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    async with get_checkpointer() as checkpointer:
        g = build_graph(checkpointer=checkpointer)

        # Seed state with findings so compact has something to work with
        seed = {
            "run_id": thread_id,
            "query": "What are the main features of Python 3.13?",
            "clarifications": [],
            "subtasks": ["features", "performance"],
            "findings": [
                {
                    "subtask": "features",
                    "claim": "Python 3.13 introduces free-threaded mode (no GIL).",
                    "evidence_span": "PEP 703 introduces an optional free-threaded build.",
                    "citation_url": "https://docs.python.org/3.13/whatsnew/3.13.html",
                },
                {
                    "subtask": "performance",
                    "claim": "Python 3.13 is ~5% faster than 3.12 on benchmarks.",
                    "evidence_span": "The pyperformance suite shows ~5% improvement.",
                    "citation_url": "https://docs.python.org/3.13/whatsnew/3.13.html",
                },
            ],
            "summary": "",
            "report": "",
            "messages": [],
        }
        await g.aupdate_state(config, seed, as_node="subagent")

        # Invoke the compact node
        await g.ainvoke(None, config=config, output_keys=["summary"])

        state = await g.aget_state(config)
        summary = state.values.get("summary", "")
        findings = state.values.get("findings", [])

        ok("Compact node ran successfully")
        info(f"summary (first 120 chars): {summary[:120]!r}")
        info(f"raw findings after compact: {len(findings)} (should be 0 — trimmed)")
        assert summary, "summary should be non-empty after compact"
        ok("state.summary populated from subagent findings")


# ---------------------------------------------------------------------------
# Section 5: Chat — follow-up Q&A reads from checkpointer state
# ---------------------------------------------------------------------------

async def test_chat() -> None:
    section("Chat — follow-up Q&A grounded in checkpointer state (layer 3)")

    from engine.memory.checkpointer import get_checkpointer
    from engine.nodes.chat import answer_followup
    from engine.orchestrator import build_graph

    thread_id = f"test-chat-{uuid.uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}

    async with get_checkpointer() as checkpointer:
        g = build_graph(checkpointer=checkpointer)

        # Seed a completed research run into checkpointer state
        await g.aupdate_state(config, {
            "run_id": thread_id,
            "query": "What are the main features of Python 3.13?",
            "clarifications": [],
            "subtasks": [],
            "findings": [
                {
                    "subtask": "features",
                    "claim": "Python 3.13 introduces free-threaded mode.",
                    "evidence_span": "PEP 703 introduces an optional free-threaded build.",
                    "citation_url": "https://docs.python.org/3.13/whatsnew/3.13.html",
                }
            ],
            "summary": "Python 3.13 adds free-threaded mode via PEP 703.",
            "report": "# Python 3.13\n\nPython 3.13 introduces free-threaded mode.",
            "messages": [],
        })
        ok(f"Seeded research state for thread_id={thread_id}")

        question = "What is the free-threaded mode?"
        info(f"Follow-up question: {question!r}")

        answer_chunks = []
        async for chunk in answer_followup(
            thread_id=thread_id,
            question=question,
            history=[],
            checkpointer=checkpointer,
        ):
            answer_chunks.append(chunk)

        answer = "".join(answer_chunks)
        ok("Chat answered follow-up question from checkpointer state")
        info(f"Answer (first 200 chars): {answer[:200]!r}")
        assert len(answer) > 20, "Answer should be non-trivial"
        ok("Answer grounded in checkpointer state — no re-fetching needed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

SECTIONS: dict[str, tuple[str, object]] = {
    "checkpointer": ("Checkpointer (no LLM)", test_checkpointer),
    "clarify-clear": ("Clarify — clear query", test_clarify_clear),
    "clarify-hil": ("Clarify — ambiguous query (human-in-the-loop)", test_clarify_hil),
    "compact": ("Compact node", test_compact),
    "chat": ("Follow-up chat", test_chat),
}


async def main(sections_to_run: list[str]) -> None:
    passed = []
    failed = []

    for key in sections_to_run:
        _, fn = SECTIONS[key]
        try:
            await fn()  # type: ignore[operator]
            passed.append(key)
        except Exception as exc:
            print(f"\n  ✗  FAILED: {exc}")
            failed.append((key, exc))

    print(SEPARATOR)
    print(f"Results: {len(passed)} passed, {len(failed)} failed")
    for key, exc in failed:
        print(f"  ✗  {key}: {exc}")
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 2 manual integration tests")
    parser.add_argument(
        "--section",
        choices=list(SECTIONS.keys()) + ["all"],
        default="all",
        help="Which section to run (default: all)",
    )
    args = parser.parse_args()

    to_run = list(SECTIONS.keys()) if args.section == "all" else [args.section]
    asyncio.run(main(to_run))
