# multiagent-deep-research-system

General-purpose multi-agent deep research system — LangGraph + FastAPI + Next.js.

- Multi-agent orchestration: plan → parallel `Send` fan-out subagents → compact → synthesize → cited report
- Three-layer memory stack: typed working state, context compaction, Postgres checkpointer
- Human-in-the-loop clarification via LangGraph `interrupt()` / resume
- **Adversarial debate mode**: two agents from different AI companies (Claude advocate vs Gemini skeptic) argue over the findings before a neutral GPT writer synthesizes the report — uncorrelated errors catch more gaps
- Anti-hallucination: per-sentence citation faithfulness verification + eval harness
