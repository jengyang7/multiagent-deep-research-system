# Issues Log

Running record of bugs, gotchas, and fixes encountered while building this project.

---

## Phase 2 — Memory + Human-in-the-loop (integration test run)

### 1. `Send` imported from deprecated location
**File:** `engine/orchestrator.py`
**Issue:** `from langgraph.constants import Send` emitted a `LangGraphDeprecatedSinceV10` warning — this import is scheduled for removal in LangGraph V2.
**Fix:** `from langgraph.types import Send`

---

### 2. `add_messages` not in `langchain_core.messages`
**File:** `engine/state.py`
**Issue:** `from langchain_core.messages import add_messages` raises `ImportError`. Despite being a message-related utility, it lives in LangGraph, not LangChain core.
**Fix:** `from langgraph.graph.message import add_messages`

---

### 3. Test failed because `ChatOpenAI.__init__` requires `OPENAI_API_KEY`
**File:** `tests/test_phase2_smoke.py` (`test_synthesize_uses_summary_over_raw_findings`)
**Issue:** The test patched `_PROMPT` to intercept the chain but still let `ChatOpenAI(...)` run, which immediately throws `OpenAIError: Missing credentials` — even in a unit test with no intent to make API calls.
**Fix:** Patch `ChatOpenAI` itself in addition to `_PROMPT`:
```python
monkeypatch.setattr("engine.nodes.synthesize.ChatOpenAI", lambda **kw: mock_llm)
```

---

### 6. Checkpointer state read empty after breaking from stream early
**File:** `scripts/test_phase2.py`
**Issue:** After calling `Command(resume=answers)` and breaking from the `astream_events` loop immediately after `clarify_wait`'s `on_chain_end` event, `aget_state()` still returned the pre-resume state (`clarifications=[]`, query unchanged). LangGraph flushes checkpoints to Postgres as part of the stream-processing loop — abandoning the generator at `on_chain_end` exits before that flush happens.

**Fix (two-part):**
1. Capture `clarify_wait`'s output directly from the `on_chain_end` event data (`event["data"]["output"]`) — this is the node's return value and is reliable regardless of checkpointer timing.
2. Continue consuming the stream until `plan`'s `on_chain_start` event — this is the signal that the previous checkpoint (for `clarify_wait`) has been written. Then break and read from `aget_state` safely.

**Rule:** When reading checkpointer state after a resume, always drain the stream at least one node past the resumed node before breaking. Breaking at the resumed node's `on_chain_end` is too early.

---

### 5. `clarify` node re-ran the LLM on resume, discarding user answers
**File:** `engine/nodes/clarify.py`, `engine/orchestrator.py`
**Issue:** When `Command(resume=answers)` resumed the graph, LangGraph re-ran the `clarify` node from the top. The LLM was called again with the same query and this time returned `is_ambiguous=False`, taking the early-return branch before ever reaching `interrupt()`. Result: `clarifications=[]`, refined query unchanged, test failed with "Expected clarifications to be populated after resume".

**Root cause:** In LangGraph, `interrupt()` pauses by raising `GraphInterrupt` — so no state updates can be saved before it. On resume, the entire node re-executes from the top. If the LLM is also in that node, it runs again with a potentially different result.

**Fix:** Split `clarify` into two nodes:
- `clarify` — calls the LLM once, stores questions in `state.clarification_questions`. Skips the LLM if questions are already stored (idempotent on retry).
- `clarify_wait` — calls `interrupt(questions)`. On first pass: raises `GraphInterrupt` (pauses). On resume: `interrupt()` returns answers immediately without re-calling the LLM.

`state.next` now points to `clarify_wait`, not `clarify`, so the LLM node is never re-entered on resume.

---

### 4. Ruff lint: unsorted imports (I001), unused import (F401), unused variable (F841), long lines (E501)
**Files:** `engine/nodes/clarify.py`, `engine/nodes/plan.py`, `engine/orchestrator.py`, `tests/test_phase2_smoke.py`, `engine/models.py`
**Issue:** Several files had import blocks out of ruff's expected order (third-party before first-party, alphabetical within groups). A `patch` import was added but never used. A `original_prompt` variable was assigned but never read. Two comment lines in `models.py` exceeded the 100-char limit.
**Fix:** `uv run ruff check --fix` resolved the import ordering, unused import, and unused variable automatically. The long comment lines in `models.py` were shortened manually.
