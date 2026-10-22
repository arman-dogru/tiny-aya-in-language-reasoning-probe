"""Pure functions: answer parsing, trace extraction, repetition, language ID helpers."""
from __future__ import annotations

import re
from collections import Counter

ANS_RE = re.compile(r"####\s*([^\n]*)")
NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def split_trace(completion: str, think_tags: bool, think_close: str = "</think>") -> tuple[str, str]:
    """Return (reasoning_trace, answer_part). The answer part starts at the last '####'."""
    text = completion
    if think_tags:
        if think_close in text:
            trace, rest = text.split(think_close, 1)
            return trace.strip(), rest
        return text.strip(), ""
    i = text.rfind("####")
    if i < 0:
        return text.strip(), ""
    return text[:i].strip(), text[i:]


def parse_answer(answer_part: str, task: str) -> str | None:
    m = ANS_RE.findall(answer_part)
    if not m:
        return None
    s = m[-1].strip()
    if task == "math":
        n = NUM_RE.findall(s.replace("$", ""))
        if not n:
            return None
        v = n[0].replace(",", "")
        try:
            f = float(v)
        except ValueError:
            return None
        return str(int(f)) if f == int(f) else str(f)
    s = s.strip("*_ .()[]:").upper()
    return s[0] if s and s[0] in "ABCD" else None


LETTER_RE = re.compile(r"(?:^|[\s*(\[])([ABCD])(?=[\s.)\]:*]|$)")


def parse_answer_lenient(completion: str, task: str) -> str | None:
    """Fallback when the model ignores the '####' format: the strict answer if present,
    else the last number (math) or the last standalone option letter (piqa) in the text."""
    _, ans = split_trace(completion, think_tags=False)
    strict = parse_answer(ans, task)
    if strict is not None:
        return strict
    if task == "math":
        n = NUM_RE.findall(completion.replace("$", ""))
        if not n:
            return None
        v = n[-1].replace(",", "").rstrip(".")
        try:
            f = float(v)
        except ValueError:
            return None
        return str(int(f)) if f == int(f) else str(f)
    m = LETTER_RE.findall(completion)
    return m[-1] if m else None


def words(text: str) -> list[str]:
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?", text.lower())


def repetition_score(text: str, n: int = 4) -> float | None:
    """Share of word n-grams that are repeats: 1 - unique/total. None if too short."""
    w = words(text)
    if len(w) < n + 4:
        return None
    grams = [tuple(w[i:i + n]) for i in range(len(w) - n + 1)]
    return 1.0 - len(set(grams)) / len(grams)


def lid_clean(text: str) -> str:
    """Strip LaTeX, math, numbers and markup before language ID."""
    t = re.sub(r"\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$", " ", text, flags=re.S)
    t = re.sub(r"\\[a-zA-Z]+", " ", t)
    t = re.sub(r"[0-9]+([.,][0-9]+)*", " ", t)
    t = re.sub(r"[=+\-*/×÷^_{}\[\]()<>|#%$€£:;]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def segments(text: str) -> list[str]:
    """Sentence-ish segments for segment-level language ID."""
    parts = re.split(r"(?<=[.!?。])\s+|\n+", text)
    return [p.strip() for p in parts if p.strip()]


class LID:
    def __init__(self, path: str):
        import fasttext  # noqa: PLC0415
        self.m = fasttext.load_model(path)

    def label(self, text: str, min_words: int = 3) -> tuple[str, float]:
        t = lid_clean(text)
        if len(words(t)) < min_words:
            return "und", 0.0
        (lab,), (p,) = self.m.predict(t.replace("\n", " "), k=1)
        return lab.replace("__label__", ""), float(min(p, 1.0))

    def segment_shares(self, text: str) -> Counter:
        """Character-weighted share of each detected language over segments with >=3 words."""
        c: Counter = Counter()
        for s in segments(text):
            lab, _ = self.label(s)
            if lab != "und":
                c[lab] += len(lid_clean(s))
        tot = sum(c.values())
        return Counter({k: v / tot for k, v in c.items()}) if tot else Counter()
