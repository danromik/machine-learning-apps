"""Phase 1 spike: confirm Claude Agent SDK can authenticate via the local
Claude Code subscription (no ANTHROPIC_API_KEY) and reach claude-opus-4-7.

Removes ANTHROPIC_API_KEY from the environment if it happens to be set, so
this proves subscription-only auth — not API-key fallback.
"""

from __future__ import annotations

import asyncio
import os

from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    query,
)

MODEL = "claude-opus-4-7[1m]"


async def main() -> None:
    # Force subscription auth — remove API key if present.
    os.environ.pop("ANTHROPIC_API_KEY", None)
    os.environ.pop("ANTHROPIC_AUTH_TOKEN", None)
    # Avoid Claude-Code-in-Claude-Code env confusion when the spike is run
    # from inside a Claude Code session.
    for k in ("CLAUDECODE", "CLAUDE_CODE_ENTRYPOINT", "CLAUDE_CODE_EXECPATH",
              "AI_AGENT", "ANTHROPIC_MODEL"):
        os.environ.pop(k, None)

    options = ClaudeAgentOptions(
        model=MODEL,
        system_prompt="You are a terse assistant. Reply in one short sentence.",
    )

    prompt = "Reply with exactly: ML Engineer ready, model=<your model id>."
    print(f"[spike] sending prompt to {MODEL!r}…")

    text_chunks: list[str] = []
    result: ResultMessage | None = None

    async for message in query(prompt=prompt, options=options):
        if isinstance(message, AssistantMessage):
            for block in message.content:
                if isinstance(block, TextBlock):
                    text_chunks.append(block.text)
        elif isinstance(message, ResultMessage):
            result = message

    print("\n--- assistant said ---")
    print("".join(text_chunks).strip() or "(no text returned)")
    print("--- result ---")
    if result is not None:
        # ResultMessage attributes vary by SDK version; print whatever's there.
        for attr in ("session_id", "duration_ms", "total_cost_usd", "usage", "is_error", "subtype"):
            if hasattr(result, attr):
                print(f"{attr}: {getattr(result, attr)}")
    else:
        print("(no ResultMessage)")


if __name__ == "__main__":
    asyncio.run(main())
