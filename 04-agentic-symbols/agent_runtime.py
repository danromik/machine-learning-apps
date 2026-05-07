"""ML Engineer agent runtime: runs a single conversational turn against
the Claude Agent SDK and yields normalized events the frontend can render.

The chat is a sequence of turns. Each turn:
  - takes a user prompt + optional session id to resume
  - sends it through `query()` with our MCP tool server attached
  - yields events: text_delta / text_message / tool_use / tool_result /
    usage / result / error

The session id is captured from the SDK's ResultMessage and sent back to
the client so it can persist it under the active chat. SDK persists the
full message history at `~/.claude/projects/<encoded-cwd>/<id>.jsonl`,
which we read back when the user resumes a past session.

Model + system prompt + tool surface are wired here so the FastAPI
endpoint stays a thin shell over `run_turn`.
"""
from __future__ import annotations

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

# Pin the model. 1M context variant — long sessions pile up tool results.
MODEL = "claude-opus-4-7[1m]"
CONTEXT_WINDOW = 1_000_000  # for the UI's progress bar

# Hard cap on tool-loop iterations per user turn. Generous for
# autonomous training runs (each train_n_batches = 1 turn) but bounded
# so a runaway loop can't burn through the user's quota silently.
MAX_TURNS_PER_USER_MESSAGE = 80


def _read_system_prompt() -> str:
    return SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def _build_options(*, resume_session_id: str | None) -> ClaudeAgentOptions:
    """Construct the options dict for one turn.

    The MCP server is rebuilt per turn — cheap (in-process Python) and
    avoids any state surviving across turns inside the server.
    """
    return ClaudeAgentOptions(
        model=MODEL,
        system_prompt=_read_system_prompt(),
        mcp_servers={"agentic-symbols": build_mcp_server()},
        allowed_tools=allowed_tool_names(),
        # Restrict to our MCP tools — the agent has no Read/Write/Bash
        # access to the host filesystem. Defense in depth: even if the
        # system prompt didn't say "don't use shell tools", the SDK
        # filter would block them.
        cwd=str(_HERE),
        resume=resume_session_id,
        include_partial_messages=True,
        max_turns=MAX_TURNS_PER_USER_MESSAGE,
    )


def _content_block_to_event(block: Any) -> dict[str, Any] | None:
    """Convert a single content block from a final assistant/user message
    into one of our normalized events. Returns None for blocks we don't
    surface (e.g. raw thinking blocks, system blocks)."""
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


def _short_tool_name(full: str) -> str:
    """`mcp__agentic-symbols__train_n_batches` → `train_n_batches`."""
    parts = full.split("__")
    return parts[-1] if parts else full


def _stringify_tool_result(content: Any) -> str:
    """Tool results come back as either a string or a list of content
    blocks. Flatten to a single string for the UI."""
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


def _stream_event_to_event(message: StreamEvent) -> dict[str, Any] | None:
    """Pull text-delta + usage updates out of the raw SDK stream events.
    Tool-use streaming is left to the final AssistantMessage — the
    deltas for tool_use blocks are messy and we don't render them
    incrementally anyway."""
    ev = message.event
    t = ev.get("type")
    if t == "content_block_delta":
        delta = ev.get("delta", {}) or {}
        if delta.get("type") == "text_delta":
            return {"type": "text_delta", "text": delta.get("text", "")}
    elif t == "message_delta":
        usage = (ev.get("usage") or {})
        if usage:
            return {
                "type": "usage",
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
                "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
            }
    return None


# ── Public API ─────────────────────────────────────────────────────────


async def run_turn(
    prompt: str, *, resume_session_id: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run one user → assistant turn. Yields a sequence of event dicts:

      {type: "text_delta", text: "…"}                    streamed text
      {type: "text_message", text: "…"}                  finalized bubble
      {type: "tool_use", id, name, input}                tool call started
      {type: "tool_result", tool_use_id, content, is_error}
      {type: "usage", input_tokens, …}                   running totals
      {type: "result", session_id, total_cost_usd, …}    end of turn
      {type: "error", message}                           something failed

    The final event of a successful run is `result`. After `error`, no
    further events are yielded.
    """
    # Subscription-auth hygiene — keep the SDK from finding a stale API
    # key in the environment, and don't let the parent Claude Code
    # session's env confuse the spawned `claude` subprocess.
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
                # Emit one structured event per content block. The text
                # ones are duplicates of the streamed deltas, but they
                # let the frontend "commit" the bubble at the right
                # boundary (between text → tool_use, tool_result → next
                # text, etc.) without parsing the raw stream itself.
                for block in message.content:
                    evt = _content_block_to_event(block)
                    if evt is not None:
                        yield evt
            elif isinstance(message, UserMessage):
                # User messages from the SDK include synthesized
                # tool_result blocks — pass those through so the
                # frontend can attach them to the matching tool_use.
                content = getattr(message, "content", None) or []
                if isinstance(content, list):
                    for block in content:
                        evt = _content_block_to_event(block)
                        if evt is not None:
                            yield evt
            elif isinstance(message, SystemMessage):
                # Skip — system messages are SDK bookkeeping (model info,
                # mcp status). Nothing to render.
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
