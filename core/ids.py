"""Short, non-identifying request ids shared by retrieval and answers."""

from __future__ import annotations

import secrets


def new_qid() -> str:
    """Return a random query id such as ``q_3f9a1c``.

    The id is deliberately random rather than a timestamp or counter: it is
    safe to show in the UI, cannot be traced back to a user, and is unique
    enough for demo-scale artifacts.
    """
    return f"q_{secrets.token_hex(3)}"


__all__ = ["new_qid"]
