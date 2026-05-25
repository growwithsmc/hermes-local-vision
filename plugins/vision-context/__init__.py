"""
Vision Context Plugin for Hermes Agent
=======================================
When `auxiliary.vision.mode: context` is set, automatically includes the last
N conversation turns as context in every auxiliary vision request.

Prevents hallucinations by letting the vision model know what conversation
it's part of before analyzing the image.

Config:
  auxiliary:
    vision:
      mode: context           # "stateless" (default) or "context"
      context_messages: 3     # last N user/assistant pairs to include

Zero core file modifications - all logic lives in this plugin.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import Any, Dict

logger = logging.getLogger("hermes.plugins.vision_context")

# ---------------------------------------------------------------------------
# Context loader
# ---------------------------------------------------------------------------

def _load_recent_context(max_pairs: int = 3, state_db_path: str = "") -> str:
    """Load the last N user/assistant message pairs from the current session.

    Returns formatted context string, or empty string if unavailable.
    """
    try:
        if not state_db_path or not os.path.exists(state_db_path):
            return ""

        conn = sqlite3.connect(state_db_path)
        conn.row_factory = sqlite3.Row
        c = conn.cursor()

        c.execute("SELECT id FROM sessions ORDER BY started_at DESC LIMIT 1")
        row = c.fetchone()
        if not row:
            conn.close()
            return ""
        session_id = row["id"]

        limit = 2 * max_pairs + 1
        c.execute(
            "SELECT role, content FROM messages "
            "WHERE session_id = ? AND role IN ('user', 'assistant') "
            "ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        )
        rows = c.fetchall()
        conn.close()

        if not rows:
            return ""

        rows.reverse()

        parts = ["[Previous conversation]"]
        for r in rows:
            role = r["role"].capitalize()
            text = str(r["content"] or "")[:2000]
            parts.append(f"{role}: {text}")
        parts.append("[End of previous conversation]")

        return "\n\n".join(parts)

    except Exception as e:
        logger.debug("vision-context: failed to load: %s", e)
        return ""


# ---------------------------------------------------------------------------
# Monkey-patch
# ---------------------------------------------------------------------------

_original_handler = None
_hermes_home = ""


def _patched_handler(args: Dict[str, Any], **kw: Any) -> Any:
    """Wrapper that adds conversation context before calling the real handler."""
    global _hermes_home

    try:
        from hermes_cli.config import cfg_get, load_config

        cfg = load_config()
        vision_cfg = cfg_get(cfg, "auxiliary", "vision", default={})
        mode = vision_cfg.get("mode", "stateless")
        context_pairs = int(vision_cfg.get("context_messages", 3))

        if mode == "context" and context_pairs > 0:
            state_db = os.path.join(_hermes_home, "state.db")
            context_str = _load_recent_context(context_pairs, state_db)
            if context_str:
                question = args.get("question", "")
                args = dict(args)
                args["question"] = (
                    f"{context_str}\n\n"
                    f"---\n\n"
                    f"{question}"
                )
                logger.info(
                    "vision-context: prepended %d chars from %d recent pairs",
                    len(context_str), context_pairs,
                )
    except Exception as e:
        logger.debug("vision-context: patch skipped: %s", e)

    return _original_handler(args, **kw)


# ---------------------------------------------------------------------------
# Plugin lifecycle
# ---------------------------------------------------------------------------

def register(ctx) -> None:
    """Called by Hermes plugin loader. Installs the monkey-patch."""
    global _original_handler, _hermes_home

    from hermes_constants import get_hermes_dir
    _hermes_home = get_hermes_dir()

    try:
        import tools.vision_tools as vt
        _original_handler = vt._handle_vision_analyze
        vt._handle_vision_analyze = _patched_handler

        logger.info(
            "vision-context: installed (mode=%s, pairs=%d)",
            ctx.manifest.config.get("mode", "context"),
            ctx.manifest.config.get("context_messages", 3),
        )

        # Register a /vision-context slash command
        @ctx.command("vision-context")
        def cmd_vision_context(args: str) -> str:
            """Show current vision context config."""
            from hermes_cli.config import cfg_get, load_config
            cfg = load_config()
            vc = cfg_get(cfg, "auxiliary", "vision", default={})
            mode = vc.get("mode", "stateless")
            pairs = vc.get("context_messages", 3)
            return (
                f"Vision context mode: {mode}\n"
                f"Context messages: {pairs}\n"
                f"State DB: {os.path.join(_hermes_home, 'state.db')}"
            )

    except Exception as e:
        logger.error("vision-context: failed to install: %s", e)
