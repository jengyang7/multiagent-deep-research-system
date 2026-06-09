"""SHORT-TERM / EPISODIC MEMORY (layer 3 of the memory stack):
LangGraph Postgres checkpointer — persists graph state per thread_id.
Enables: resumable runs, human-in-the-loop pause/resume, and multi-turn
follow-up chat (all read from the same persisted state snapshot).
"""
from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver


def _conn_string() -> str:
    from urllib.parse import parse_qs, urlencode, urlparse, urlunparse
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL environment variable is not set")
    # Convert scheme: asyncpg → psycopg (sync, used by checkpointer)
    url = url.replace("postgresql+asyncpg://", "postgresql://")
    # Normalise SSL param: asyncpg uses ssl=, psycopg uses sslmode=
    parsed = urlparse(url)
    params = {k: v[0] for k, v in parse_qs(parsed.query).items()}
    ssl_val = params.pop("ssl", params.pop("sslmode", None))
    for unsupported in ("channel_binding", "options"):
        params.pop(unsupported, None)
    if ssl_val:
        params["sslmode"] = ssl_val
    return urlunparse(parsed._replace(query=urlencode(params)))


@asynccontextmanager
async def get_checkpointer() -> AsyncIterator[AsyncPostgresSaver]:
    """Async context manager that yields a ready AsyncPostgresSaver."""
    async with AsyncPostgresSaver.from_conn_string(_conn_string()) as checkpointer:
        await checkpointer.setup()
        yield checkpointer
