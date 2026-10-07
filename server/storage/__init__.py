"""Storage helpers for persisting slide state and history."""

from .manager import (
    DEFAULT_SESSION_ID,
    SlideState,
    append_slide_history,
    ensure_session,
    load_last_state,
    load_slide_history,
    log_qa_turn,
    reset_slide_history,
    save_last_state,
)

__all__ = [
    "DEFAULT_SESSION_ID",
    "SlideState",
    "append_slide_history",
    "ensure_session",
    "load_last_state",
    "load_slide_history",
    "log_qa_turn",
    "reset_slide_history",
    "save_last_state",
]
