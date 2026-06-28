"""RL Coach agent runtime: runs a single conversational turn against the
Claude Agent SDK and yields normalized events the frontend can render.

Ported from 04-agentic-symbols' agent_runtime — the streaming/normalization
logic is domain-agnostic. The only RL-specific wiring is the MCP server name
and tool surface (from agent_tools) and the system prompt.

Each turn takes a user prompt + optional session id to resume, sends it
through `query()` with our in-process MCP tool server attached, and yields
events: text_delta / text_message / tool_use / tool_result / usage / result /
error. The SDK persists full history at
`~/.claude/projects/<encoded-cwd>/<id>.jsonl`, read back on resume.
"""
from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, AsyncIterator

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    StreamEvent,
    SystemMessage,
    TextBlock,
    ToolResultBlock,
    ToolUseBlock,
    UserMessage,
    query,
)

from agent_tools import allowed_tool_names, build_mcp_server

_HERE = Path(__file__).resolve().parent
SYSTEM_PROMPT_PATH = _HERE / "agent_system_prompt.md"
MCP_SERVER_NAME = "agentic-cube"

# 1M-context variant — long coaching sessions accumulate tool results.
MODEL = "claude-opus-4-8[1m]"
CONTEXT_WINDOW = 1_000_000  # for the UI's progress bar

# Generous cap on tool-loop iterations per user turn (each train tool call is
# one turn), bounded so a runaway loop can't silently burn quota.
MAX_TURNS_PER_USER_MESSAGE = 80

# Serializes agent turns so a user chat turn and an autonomous check-in never
# interleave on the same resumed coach session. Both the SSE chat handler and the
# check-in scheduler acquire this; a user message always wins (the scheduler
# defers when it's locked).
AGENT_TURN_LOCK = asyncio.Lock()


def _read_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _build_options(*, resume_session_id: str | None) -> ClaudeAgentOptions:
    return ClaudeAgentOptions(
        model=MODEL,
        system_prompt=_read_system_prompt(),
        mcp_servers={MCP_SERVER_NAME: build_mcp_server()},
        allowed_tools=allowed_tool_names(),
        cwd=str(_HERE),
        resume=resume_session_id,
        include_partial_messages=True,
        max_turns=MAX_TURNS_PER_USER_MESSAGE,
    )


def _short_tool_name(full: str) -> str:
    parts = full.split("__")
    return parts[-1] if parts else full


def _stringify_tool_result(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out: list[str] = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                out.append(str(b.get("text", "")))
            elif isinstance(b, str):
                out.append(b)
            else:
                out.append(str(b))
        return "\n".join(out)
    return str(content)


def _content_block_to_event(block: Any) -> dict[str, Any] | None:
    if isinstance(block, TextBlock):
        return {"type": "text_message", "text": block.text}
    if isinstance(block, ToolUseBlock):
        return {
            "type": "tool_use",
            "id": block.id,
            "name": _short_tool_name(block.name),
            "full_name": block.name,
            "input": block.input,
        }
    if isinstance(block, ToolResultBlock):
        return {
            "type": "tool_result",
            "tool_use_id": block.tool_use_id,
            "content": _stringify_tool_result(block.content),
            "is_error": bool(getattr(block, "is_error", False)),
        }
    return None


def _stream_event_to_event(message: StreamEvent) -> dict[str, Any] | None:
    ev = message.event
    t = ev.get("type")
    if t == "content_block_delta":
        delta = ev.get("delta", {}) or {}
        if delta.get("type") == "text_delta":
            return {"type": "text_delta", "text": delta.get("text", "")}
    elif t == "message_delta":
        usage = ev.get("usage") or {}
        if usage:
            return {
                "type": "usage",
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            }
    return None


async def run_turn(
    prompt: str, *, resume_session_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run one user → assistant turn, yielding normalized event dicts. The
    final event of a successful run is `result`; after `error`, nothing more."""
    # Keep the SDK from finding a stale API key / the parent Claude Code
    # session's env in the spawned subprocess.
    for k in (
        "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN",
        "CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT",
        "CLAUDE_CODE_EXECPATH", "AI_AGENT", "ANTHROPIC_MODEL",
    ):
        os.environ.pop(k, None)

    options = _build_options(resume_session_id=resume_session_id)

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, StreamEvent):
                evt = _stream_event_to_event(message)
                if evt is not None:
                    yield evt
            elif isinstance(message, AssistantMessage):
                for block in message.content:
                    evt = _content_block_to_event(block)
                    if evt is not None:
                        yield evt
            elif isinstance(message, UserMessage):
                content = getattr(message, "content", None) or []
                if isinstance(content, list):
                    for block in content:
                        evt = _content_block_to_event(block)
                        if evt is not None:
                            yield evt
            elif isinstance(message, SystemMessage):
                continue
            elif isinstance(message, ResultMessage):
                yield {
                    "type": "result",
                    "session_id": getattr(message, "session_id", None),
                    "total_cost_usd": getattr(message, "total_cost_usd", None),
                    "duration_ms": getattr(message, "duration_ms", None),
                    "is_error": bool(getattr(message, "is_error", False)),
                    "subtype": getattr(message, "subtype", None),
                    "usage": getattr(message, "usage", None),
                }
                return
    except Exception as e:
        yield {"type": "error", "message": f"{type(e).__name__}: {e}"}
        return
