"""Input and output guardrails (stage v5+). Cheap, deterministic checks that run around the model."""

from __future__ import annotations

import re
from dataclasses import dataclass

MAX_INPUT_CHARS = 1000

_INJECTION = re.compile("|".join([
    r"ignore (all |any |the |your )?(previous |prior |above |earlier )?(instructions|prompts?|rules)",
    r"(reveal|show|print|repeat|leak) (me )?(your|the) (system |hidden )?(prompt|instructions)",
    r"\byou are now\b", r"\bjailbreak\b", r"\bdeveloper mode\b", r"\bact as (an? )?(unrestricted|unfiltered)",
    r"oublie (toutes |tes |les )?(tes |les )?(instructions|règles)", r"ignore (tes|les) (instructions|règles)",
]), re.I)
_SMALLTALK = re.compile(
    r"^\s*(hi|hello|hey|salam|salut|bonjour|bonsoir|good (morning|afternoon|evening)|thanks?( you)?|merci|bye|goodbye|"
    r"au revoir|who are you|what are you|what can you do|qui es[- ]tu|comment (ça|ca) va|how are you|"
    r"my name is|je m'appelle)\b", re.I)
_ABOUT_CONVERSATION = re.compile(
    r"\b(my name|mon nom|i told you|i said|you said|earlier|previous(ly)?|je t'ai dit|tu as dit|tout à l'heure)\b", re.I)
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE = re.compile(r"(?<!\d)(?:\+?\d[\d .-]{7,}\d)(?!\d)")
_CITATION = re.compile(r"\[(\d+)\]")

REFUSALS = {
    "too_long": "That message is too long for me. Could you ask your question in a sentence or two?",
    "prompt_injection": "I can't change my instructions, but I'm happy to help with questions about EMINES and the campus.",
    "empty": "I didn't catch that. Could you repeat your question?",
    "no_grounding": "I couldn't find that in the EMINES documents I rely on, so I'd rather not guess. "
                    "The reception desk in the main hall can help you.",
}


@dataclass(frozen=True)
class Verdict:
    allowed: bool
    reason: str | None = None

    @property
    def message(self) -> str | None:
        return REFUSALS.get(self.reason) if self.reason else None


def check_input(text: str) -> Verdict:
    if not text.strip():
        return Verdict(False, "empty")
    if len(text) > MAX_INPUT_CHARS:
        return Verdict(False, "too_long")
    if _INJECTION.search(text):
        return Verdict(False, "prompt_injection")
    return Verdict(True)


def is_smalltalk(text: str) -> bool:
    return bool(_SMALLTALK.search(text))


def refers_to_conversation(text: str) -> bool:
    """Questions answered from the chat history, not the documents ("what's my name?")."""
    return bool(_ABOUT_CONVERSATION.search(text))


def redact_pii(text: str) -> str:
    """Applied to everything written to the trace store: logs must never hold contact details."""
    def phone(match: re.Match) -> str:  # 9+ digits, so dates and times are left alone
        return "[PHONE]" if sum(ch.isdigit() for ch in match.group(0)) >= 9 else match.group(0)

    return _PHONE.sub(phone, _EMAIL.sub("[EMAIL]", text))


def check_output(answer: str, n_sources: int) -> tuple[str, list[str]]:
    """Drop citations to sources that were never retrieved and flag uncited grounded answers."""
    issues: list[str] = []

    def fix(match: re.Match) -> str:
        if 1 <= int(match.group(1)) <= n_sources:
            return match.group(0)
        issues.append(f"invalid_citation_{match.group(1)}")
        return ""

    cleaned = _CITATION.sub(fix, answer).strip()
    if n_sources and not _CITATION.search(cleaned):
        issues.append("uncited")
    if not cleaned:
        issues.append("empty_answer")
        cleaned = REFUSALS["empty"]
    return cleaned, issues


def cited_numbers(answer: str) -> list[int]:
    return sorted({int(n) for n in _CITATION.findall(answer)})
