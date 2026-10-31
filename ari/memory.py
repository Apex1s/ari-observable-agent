"""Conversation memory (stage v4+): a sliding window of recent turns per session."""

from __future__ import annotations

import threading
from collections import defaultdict


class ConversationMemory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max_turns
        self._sessions: dict[str, list[dict[str, str]]] = defaultdict(list)
        self._lock = threading.Lock()

    def history(self, session_id: str) -> list[dict[str, str]]:
        with self._lock:
            return list(self._sessions[session_id][-2 * self.max_turns:])

    def append(self, session_id: str, user: str, assistant: str) -> None:
        with self._lock:
            turns = self._sessions[session_id]
            turns += [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]
            del turns[:-2 * self.max_turns]

    def reset(self, session_id: str | None = None) -> None:
        with self._lock:
            if session_id is None:
                self._sessions.clear()
            else:
                self._sessions.pop(session_id, None)
