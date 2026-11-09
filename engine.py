"""Paraphrasing engine: back-translation plus conservative wording cleanup.

Pure Python with no Streamlit imports, so it can be unit-tested offline.

Pipeline
    sanitize -> protect (URLs, citations, numbers) -> back-translate in chunks
    -> quality guard per chunk -> wording / tone / punctuation cleanup -> restore
"""
from __future__ import annotations

import difflib
import functools
import html
import random
import re
import threading
import time
import unicodedata
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable, Optional

from deep_translator import GoogleTranslator

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
MAX_INPUT_CHARS = 20_000
CHUNK_CHARS = 1_500            # Google's limit is 5,000; stay well below it
MAX_WORKERS = 4                # parallel chunks; higher values trigger rate limits
MIN_WORD_SIMILARITY = 0.30     # below this a round trip is treated as meaning drift
LENGTH_RATIO_RANGE = (0.5, 1.8)

PIVOT_LANGUAGES = {
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "pt": "Portuguese",
    "it": "Italian",
    "nl": "Dutch",
}
MODE_HOPS = {"light": 0, "balanced": 1, "strong": 2}
TONES = ("academic", "neutral", "conversational")


class TranslationError(RuntimeError):
    """The translation service did not return a usable result."""


@dataclass
class Options:
    mode: str = "balanced"                 # light | balanced | strong
    tone: str = "academic"                 # academic | neutral | conversational
    strip_invisible: bool = True
    strip_chatbot: bool = True
    fix_dashes: bool = True
    replace_wording: bool = True
    pivots: tuple = tuple(PIVOT_LANGUAGES)
    seed: Optional[int] = None


@dataclass
class Result:
    text: str
    source: str
    warnings: list[str] = field(default_factory=list)
    stats: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# Sanitising
# --------------------------------------------------------------------------- #
_INVISIBLE = re.compile("[­​-‏‪-‮⁠-⁤⁦-⁩﻿︀-️]")
_CONTROL = re.compile("[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_ODD_SPACES = re.compile("[   -   　]")
_LIGATURES = str.maketrans({"ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl"})


def sanitize(text: str, strip_invisible: bool = True) -> str:
    """Normalise line endings, remove hidden characters, tidy whitespace.

    Uses NFC (not NFKC/NFKD) so superscripts, accents and symbols such as
    m² or é are preserved.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if strip_invisible:
        text = _INVISIBLE.sub("", text)
    text = _CONTROL.sub("", text)
    text = _ODD_SPACES.sub(" ", text)
    text = unicodedata.normalize("NFC", text).translate(_LIGATURES)
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    return re.sub(r"\n{3,}", "\n\n", "\n".join(lines)).strip()


def fix_dashes(text: str) -> str:
    """Replace em dashes, spaced en dashes and spaced double hyphens with commas.

    Unspaced en dashes (2010–2020) are numeric ranges and are left alone.
    """
    return re.sub(r"[ \t]*—[ \t]*|[ \t]+--+[ \t]+|[ \t]+–[ \t]+", ", ", text)


def strip_chatbot_wrapper(text: str) -> str:
    """Drop whole-line chatbot preambles and sign-offs ("Certainly!", "Hope this helps")."""
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    if lines:
        first = lines[0].strip()
        bare = first.lower().rstrip("!.,: ")
        if len(first) < 110 and (
            bare in {"certainly", "sure", "of course", "absolutely"}
            or (re.match(r"(?i)^(certainly|sure|of course|absolutely)\b", first) and first.endswith((":", "!")))
            or re.match(r"(?i)^here(?:'s| is| are)\b.{0,100}:$", first)
        ):
            lines.pop(0)
    while lines and re.match(
        r"(?i)^\s*(i hope this helps|hope this helps|let me know if|feel free to)", lines[-1]
    ):
        lines.pop()
    return "\n".join(lines).strip()


# --------------------------------------------------------------------------- #
# Protected spans (never sent to the translator)
# --------------------------------------------------------------------------- #
_PROTECT = re.compile(
    "|".join(
        [
            r"https?://[^\s<>\"')\]]*[^\s<>\"')\].,;:!?]",
            r"www\.[^\s<>\"')\]]*[^\s<>\"')\].,;:!?]",
            r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+",
            r"\b10\.\d{4,9}/[^\s]*[^\s.,;:)]",                                     # DOI
            r"\([^()\n]{0,120}\b(?:19|20)\d{2}[a-z]?\b[^()\n]{0,120}\)",           # (Author, 2020)
            r"\[\d+(?:\s*[-–,]\s*\d+)*\]",                                     # [1], [2-4]
            r"(?:[$€£₹]\s?)?(?<![A-Za-z])(?:\d[\d,.]*\d%?|\d%)",     # numbers, %, money
        ]
    )
)
_PLACEHOLDER = re.compile(r"QX\s?(\d+)\s?XQ", re.I)


def protect(text: str) -> tuple[str, list[str]]:
    store: list[str] = []

    def _sub(m: re.Match) -> str:
        store.append(m.group(0))
        return f"QX{len(store) - 1}XQ"

    return _PROTECT.sub(_sub, text), store


def restore(text: str, store: list[str]) -> str:
    def _sub(m: re.Match) -> str:
        i = int(m.group(1))
        return store[i] if i < len(store) else m.group(0)

    return _PLACEHOLDER.sub(_sub, text)


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
_ABBREVIATIONS = ("e.g.", "i.e.", "et al.", "vs.", "fig.", "eq.", "dr.", "mr.", "mrs.", "ms.", "prof.", "approx.", "etc.", "cf.")
_SENT_RE = re.compile(r'.+?(?:[.!?]+["”’)\]]*(?=\s+["“‘(\[]?[A-Z0-9])|$)', re.S)
_LINE_PREFIX = re.compile(r"^((?:[-*•]|\d+[.)]|#{1,6})\s+)(.*)$")


def split_sentences(line: str) -> list[str]:
    merged: list[str] = []
    for m in _SENT_RE.finditer(line):
        s = m.group(0).strip()
        if not s:
            continue
        if merged and merged[-1].lower().endswith(_ABBREVIATIONS):
            merged[-1] += " " + s
        else:
            merged.append(s)
    return merged


def _hard_split(piece: str, limit: int) -> list[str]:
    if len(piece) <= limit:
        return [piece]
    out, cur = [], ""
    for word in piece.split(" "):
        if cur and len(cur) + 1 + len(word) > limit:
            out.append(cur)
            cur = word
        else:
            cur = f"{cur} {word}".strip()
    if cur:
        out.append(cur)
    return out


def make_chunks(body: str, limit: int = CHUNK_CHARS) -> list[str]:
    chunks: list[str] = []
    cur = ""
    for sentence in split_sentences(body):
        for piece in _hard_split(sentence, limit):
            if cur and len(cur) + 1 + len(piece) > limit:
                chunks.append(cur)
                cur = piece
            else:
                cur = f"{cur} {piece}".strip()
    if cur:
        chunks.append(cur)
    return chunks


# --------------------------------------------------------------------------- #
# Translation
# --------------------------------------------------------------------------- #
@functools.lru_cache(maxsize=4096)
def _hop(text: str, src: str, tgt: str) -> str:
    """One translation call with retries. Failures are not cached."""
    last: Exception | None = None
    for attempt in range(3):
        try:
            out = GoogleTranslator(source=src, target=tgt).translate(text)
            if out and out.strip():
                return out
            last = TranslationError("empty response")
        except Exception as exc:  # network, rate limit, bad payload
            last = exc
        if attempt < 2:
            time.sleep(0.5 * (2 ** attempt))
    raise TranslationError(str(last))


def _round_trip(chunk: str, pivots: tuple) -> str:
    chain = ["en", *pivots, "en"]
    cur = chunk
    for src, tgt in zip(chain, chain[1:]):
        cur = _hop(cur, src, tgt)
    return cur


def word_similarity(a: str, b: str) -> float:
    ta = re.findall(r"\w+", a.lower())
    tb = re.findall(r"\w+", b.lower())
    if not ta and not tb:
        return 1.0
    return difflib.SequenceMatcher(None, ta, tb, autojunk=False).ratio()


def _accept(orig: str, cand: str) -> bool:
    """Reject round trips that lost protected spans or drifted too far in meaning."""
    if not cand or not cand.strip():
        return False
    want = sorted(re.findall(r"QX(\d+)XQ", orig))
    got = sorted(m.group(1) for m in _PLACEHOLDER.finditer(cand))
    if want != got:
        return False
    lo, hi = LENGTH_RATIO_RANGE
    if not lo <= len(cand) / max(1, len(orig)) <= hi:
        return False
    return word_similarity(orig, cand) >= MIN_WORD_SIMILARITY


def _align_edges(orig: str, cand: str) -> str:
    cand = _PLACEHOLDER.sub(lambda m: f"QX{m.group(1)}XQ", cand).strip()
    if orig[:1].isupper() and cand[:1].islower():
        cand = cand[0].upper() + cand[1:]
    if orig[-1:] in ".!?" and cand[-1:] not in ".!?\"”’)":
        cand += orig[-1]
    return cand


def _process_chunk(chunk: str, pivots: tuple, abort: threading.Event) -> tuple[str, str]:
    if abort.is_set():
        return chunk, "skipped"
    attempts = [pivots] + ([pivots[:1]] if len(pivots) > 1 else [])
    for n, path in enumerate(attempts):
        try:
            cand = _round_trip(chunk, path)
        except TranslationError:
            return chunk, "failed"
        if _accept(chunk, cand):
            return _align_edges(chunk, cand), ("ok" if n == 0 else "degraded")
    return chunk, "rejected"


def _translate_document(text: str, pivots: tuple, progress: Optional[Callable[[float], None]]):
    lines = text.split("\n")
    layout: list = []
    jobs: list = []
    for i, line in enumerate(lines):
        m = _LINE_PREFIX.match(line)
        prefix, body = (m.group(1), m.group(2)) if m else ("", line)
        if len(body.split()) < 4:           # headings and labels pass through
            layout.append(None)
            continue
        chunks = make_chunks(body)
        layout.append((prefix, chunks))
        jobs.extend((i, j, c) for j, c in enumerate(chunks))

    results: dict = {}
    counts: Counter = Counter()
    abort = threading.Event()
    if jobs:
        with ThreadPoolExecutor(max_workers=min(MAX_WORKERS, len(jobs))) as pool:
            futures = {pool.submit(_process_chunk, c, pivots, abort): (i, j, c) for i, j, c in jobs}
            for done, fut in enumerate(as_completed(futures), start=1):
                i, j, c = futures[fut]
                try:
                    out, status = fut.result()
                except Exception:
                    out, status = c, "failed"
                results[(i, j)] = out
                counts[status] += 1
                if counts["failed"] >= 3 and counts["ok"] + counts["degraded"] == 0:
                    abort.set()          # service is down: stop burning retries
                if progress:
                    progress(done / len(jobs))

    out_lines = []
    for i, line in enumerate(lines):
        if layout[i] is None:
            out_lines.append(line)
        else:
            prefix, chunks = layout[i]
            out_lines.append(prefix + " ".join(results[(i, j)] for j in range(len(chunks))))
    return "\n".join(out_lines), counts


# --------------------------------------------------------------------------- #
# Wording, tone and punctuation
# --------------------------------------------------------------------------- #
# Only substitutions that are grammatical in any context. Terms with real
# technical meanings (e.g. "robust" standard errors) are deliberately excluded.
_SIMPLE = {
    "utilize": "use", "utilizes": "uses", "utilized": "used", "utilizing": "using",
    "utilise": "use", "utilises": "uses", "utilised": "used", "utilising": "using",
    "commence": "begin", "commences": "begins", "commenced": "began", "commencing": "beginning",
    "delve into": "explore", "delves into": "explores", "delved into": "explored", "delving into": "exploring",
    "leverages": "uses", "leveraged": "used", "leveraging": "using",
    "ascertain": "determine", "ascertains": "determines", "ascertained": "determined",
    "crucial": "important", "pivotal": "key",
    "comprehensive": "thorough", "seamless": "smooth", "seamlessly": "smoothly",
    "cutting-edge": "advanced", "underscores": "highlights", "subsequently": "later",
    "a myriad of": "many", "a plethora of": "many", "a multitude of": "many",
    "a large number of": "many",
    "in order to": "to", "due to the fact that": "because", "in the event that": "if",
    "prior to": "before", "subsequent to": "after", "in spite of the fact that": "although",
    "at this point in time": "now", "in close proximity to": "near",
    "with the exception of": "except for", "in the realm of": "in",
}
_SIMPLE_RE = re.compile(
    r"(?:\b(a|an)\s+)?\b(" + "|".join(re.escape(k) for k in sorted(_SIMPLE, key=len, reverse=True)) + r")\b",
    re.I,
)
_TRANSITIONS = {
    "moreover": "Also", "furthermore": "In addition", "additionally": "Also",
    "consequently": "As a result", "nevertheless": "Even so", "nonetheless": "Still",
}
_TRANSITION_RE = re.compile(
    r"(^|[.!?][\"”’)]?\s+)(" + "|".join(_TRANSITIONS) + r"),", re.I | re.M
)
_FILLER_RE = re.compile(
    r"\b(it is (?:important|worth) (?:to note|noting|mentioning) that|it should be noted that|"
    r"it goes without saying that|needless to say,?)\s+(\w)",
    re.I,
)


def _article(word: str) -> str:
    w = word.lower()
    if w.startswith(("hour", "honest", "honor", "heir")):
        return "an"
    if w.startswith(("uni", "use", "usu", "eu", "one", "once")):
        return "a"
    return "an" if w[0] in "aeiou" else "a"


def _case(src: str, new: str) -> str:
    return new[0].upper() + new[1:] if src[:1].isupper() else new


def replace_wording(text: str) -> str:
    def _simple(m: re.Match) -> str:
        art, word = m.group(1), m.group(2)
        new = _case(word, _SIMPLE[word.lower()])
        if art:
            a = _article(new)
            return f"{a.capitalize() if art[0].isupper() else a} {new}"
        return new

    text = _SIMPLE_RE.sub(_simple, text)
    text = _TRANSITION_RE.sub(lambda m: m.group(1) + _TRANSITIONS[m.group(2).lower()] + ",", text)
    text = _FILLER_RE.sub(lambda m: m.group(2).upper() if m.group(1)[0].isupper() else m.group(2), text)
    return text


_CONTRACTIONS = {
    "do not": "don't", "does not": "doesn't", "did not": "didn't", "is not": "isn't",
    "are not": "aren't", "was not": "wasn't", "were not": "weren't", "has not": "hasn't",
    "have not": "haven't", "had not": "hadn't", "will not": "won't", "would not": "wouldn't",
    "could not": "couldn't", "should not": "shouldn't", "cannot": "can't",
    "it is": "it's", "that is": "that's", "there is": "there's",
}
_EXPAND_NT = re.compile(
    r"\b(do|does|did|is|are|was|were|has|have|had|would|could|should|must|need|might)n['’]t\b", re.I
)
_EXPAND_SUFFIX = re.compile(r"\b(\w+)['’](re|ve|ll|m)\b", re.I)
_SUFFIX_WORDS = {"re": " are", "ve": " have", "ll": " will", "m": " am"}


def _expand(text: str) -> str:
    text = re.sub(r"\bcan['’]t\b", lambda m: _case(m.group(0), "cannot"), text, flags=re.I)
    text = re.sub(r"\bwon['’]t\b", lambda m: _case(m.group(0), "will not"), text, flags=re.I)
    text = _EXPAND_NT.sub(lambda m: m.group(1) + " not", text)
    return _EXPAND_SUFFIX.sub(lambda m: m.group(1) + _SUFFIX_WORDS[m.group(2).lower()], text)


def _contract(text: str) -> str:
    # The lookahead keeps "what it is." from becoming "what it's."
    for full, short in _CONTRACTIONS.items():
        text = re.sub(
            rf"\b{re.escape(full)}\b(?=\s+[\w\"“(])",
            lambda m, s=short: _case(m.group(0), s),
            text,
            flags=re.I,
        )
    return text


def apply_tone(text: str, tone: str) -> str:
    if tone == "academic":
        return _expand(text)
    if tone == "conversational":
        return _contract(text)
    return text


def tidy_punctuation(text: str) -> str:
    text = re.sub(r"([,;])(?=[A-Za-z])", r"\1 ", text)
    text = re.sub(r"[ ]+([,.;:!?])(?=[ ]|$|[\"”’)\]])", r"\1", text, flags=re.M)
    text = re.sub(r",(?:[ ]*,)+", ",", text)
    text = re.sub(r"(?<!\.)\.\.(?!\.)", ".", text)
    text = re.sub(r"\([ ]+", "(", text)
    text = re.sub(r"[ ]+\)", ")", text)
    return re.sub(r"[ ]{2,}", " ", text)


def postprocess(text: str, opts: Options) -> str:
    text = re.sub(r"[ \t]+", " ", text)
    if opts.fix_dashes:
        text = fix_dashes(text)
    if opts.replace_wording:
        text = replace_wording(text)
    text = apply_tone(text, opts.tone)
    return tidy_punctuation(text)


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def paraphrase(text: str, opts: Optional[Options] = None, progress: Optional[Callable[[float], None]] = None) -> Result:
    opts = opts or Options()
    if opts.mode not in MODE_HOPS:
        raise ValueError(f"Unknown mode: {opts.mode}")
    if not text or not text.strip():
        raise ValueError("Please enter some text.")
    if len(text) > MAX_INPUT_CHARS:
        raise ValueError(f"Text is too long ({len(text):,} characters). The limit is {MAX_INPUT_CHARS:,}.")

    started = time.perf_counter()
    warnings: list[str] = []
    rng = random.Random(opts.seed)

    cleaned = sanitize(text, opts.strip_invisible)
    if opts.strip_chatbot:
        cleaned = strip_chatbot_wrapper(cleaned)
    if opts.fix_dashes:
        cleaned = fix_dashes(cleaned)
    if not cleaned:
        raise ValueError("Nothing is left to paraphrase after cleanup.")

    protected, store = protect(cleaned)

    pivots: tuple = ()
    counts: Counter = Counter()
    hops = MODE_HOPS[opts.mode]
    available = [c for c in opts.pivots if c in PIVOT_LANGUAGES]
    if hops and len(available) < hops:
        if available:
            warnings.append(f"Only {len(available)} pivot language(s) selected; using a shorter path.")
            hops = len(available)
        else:
            warnings.append("No pivot language selected; applied rule-based cleanup only.")
            hops = 0
    if hops:
        pivots = tuple(rng.sample(available, hops))
        protected, counts = _translate_document(protected, pivots, progress)
        failed = counts["failed"] + counts["skipped"]
        total = sum(counts.values())
        if failed:
            warnings.append(
                f"{failed} of {total} text block(s) could not be translated (service unreachable or "
                "rate-limited) and were kept as written. Try again in a minute."
            )
        if counts["rejected"]:
            warnings.append(
                f"{counts['rejected']} block(s) were kept as written because the round trip changed "
                "the meaning too much."
            )
        if counts["degraded"]:
            warnings.append(f"{counts['degraded']} block(s) used a shorter translation path.")
    elif progress:
        progress(1.0)

    final = restore(postprocess(protected, opts), store).strip()

    similarity = word_similarity(cleaned, final)
    stats = {
        "words_in": len(cleaned.split()),
        "words_out": len(final.split()),
        "changed_pct": round((1 - similarity) * 100, 1),
        "seconds": round(time.perf_counter() - started, 1),
        "path": " → ".join(["English", *[PIVOT_LANGUAGES[p] for p in pivots], "English"]) if pivots else "Rules only",
        "blocks": sum(counts.values()),
    }
    return Result(text=final, source=cleaned, warnings=warnings, stats=stats)


def diff_html(a: str, b: str, max_tokens: int = 6000) -> Optional[str]:
    """Word-level diff as escaped HTML (<del>/<ins>). Returns None if the text is too large."""
    tok = re.compile(r"\s+|\w+|[^\w\s]")
    ta, tb = tok.findall(a), tok.findall(b)
    if len(ta) + len(tb) > max_tokens * 2:
        return None
    parts: list[str] = []
    for op, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ta, tb, autojunk=False).get_opcodes():
        old, new = html.escape("".join(ta[i1:i2])), html.escape("".join(tb[j1:j2]))
        if op == "equal":
            parts.append(old)
        else:
            if old.strip():
                parts.append(f"<del>{old}</del>")
            if new.strip():
                parts.append(f"<ins>{new}</ins>")
    return "".join(parts).replace("\n", "<br>")
