"""Instruction arms: pure functions from the correct instruction to the one the policy gets."""

import hashlib
from functools import cache

from vlashift.suites import instructions, locate, read_data


def keep(text: str) -> str:
    return text


def blank(text: str) -> str:
    return ""


def other(text: str) -> str:
    """Instruction of the next task of the same suite, wrapping around after the last one."""
    suite, position = locate(text)
    texts = instructions()[suite]
    return texts[(position + 1) % len(texts)]


def shuffled(text: str) -> str:
    """Same words in a fixed order that depends only on the text (sorted by SHA-256 keys)."""
    words = text.split()
    for salt in range(100):
        keys = [hashlib.sha256(f"{salt}|{text}|{i}".encode()).digest() for i in range(len(words))]
        candidate = " ".join(word for _, word in sorted(zip(keys, words, strict=True)))
        if candidate != text:
            return candidate
    raise ValueError(f"cannot reorder the words of {text!r}")


@cache
def paraphrases() -> dict[str, str]:
    return read_data("paraphrases.json")


def paraphrase(text: str) -> str:
    try:
        return paraphrases()[text]
    except KeyError:
        raise KeyError(f"no paraphrase stored for {text!r}") from None


_ARMS = {
    "keep": keep,
    "blank": blank,
    "other": other,
    "shuffled": shuffled,
    "paraphrase": paraphrase,
}
MODES = tuple(_ARMS)


def transform(mode: str, texts: list[str]) -> list[str]:
    try:
        arm = _ARMS[mode]
    except KeyError:
        raise ValueError(f"unknown instruction mode {mode!r}, expected one of {MODES}") from None
    return [arm(text) for text in texts]
