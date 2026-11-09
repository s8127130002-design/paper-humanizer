import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import engine  # noqa: E402


def fake_hop(text, src, tgt):
    """Stand-in translator: deterministic word swaps, keeps placeholders intact."""
    return text.replace("big", "large").replace("show", "demonstrate")


def test_sanitize_removes_hidden_chars_and_keeps_symbols():
    out = engine.sanitize("Area:​ 5 m² caféﬁx")
    assert out == "Area: 5 m² caféfix"


def test_fix_dashes_keeps_numeric_ranges():
    assert engine.fix_dashes("one — two") == "one, two"
    assert engine.fix_dashes("one—two") == "one, two"
    assert engine.fix_dashes("2010–2020") == "2010–2020"
    assert "\n" in engine.fix_dashes("end—\nnext")


def test_protect_restore_roundtrip():
    src = "See https://x.org/a?b=1. Smith et al. (Smith, 2020) found 12.5% [3] at $1,200."
    prot, store = engine.protect(src)
    assert "https" not in prot and "2020" not in prot and "12.5" not in prot
    assert engine.restore(prot, store) == src


def test_wording_articles_and_transitions():
    assert engine.replace_wording("It is a crucial step.") == "It is an important step."
    assert engine.replace_wording("a cutting-edge method") == "an advanced method"
    assert engine.replace_wording("Moreover, it works.") == "Also, it works."
    assert engine.replace_wording("We did this in order to learn.") == "We did this to learn."
    assert engine.replace_wording("It is important to note that data vary.") == "Data vary."
    # technical term left alone
    assert "robust standard errors" in engine.replace_wording("We use robust standard errors.")


def test_tone():
    assert engine.apply_tone("It doesn't work and we can't go.", "academic") == "It does not work and we cannot go."
    assert engine.apply_tone("I do not know what it is.", "conversational") == "I don't know what it is."


def test_tidy_punctuation():
    assert engine.tidy_punctuation("word , next ;ok") == "word, next; ok"


def test_chunks_respect_limit_and_keep_words():
    text = " ".join(f"Sentence number {i} says something useful." for i in range(200))
    chunks = engine.make_chunks(text, limit=300)
    assert all(len(c) <= 300 for c in chunks)
    assert " ".join(chunks).split() == text.split()


def test_chatbot_wrapper():
    assert engine.strip_chatbot_wrapper("Certainly!\nBody here.\nI hope this helps!") == "Body here."


def test_end_to_end_with_stub(monkeypatch):
    monkeypatch.setattr(engine, "_hop", fake_hop)
    text = "The big effect is shown in Table 2 (Lee, 2019). It rose by 12.5% at https://x.org/d. We show more."
    res = engine.paraphrase(text, engine.Options(mode="balanced", seed=1))
    assert "(Lee, 2019)" in res.text and "12.5%" in res.text and "https://x.org/d" in res.text
    assert "large" in res.text and not res.warnings


def test_rejected_chunk_falls_back(monkeypatch):
    monkeypatch.setattr(engine, "_hop", lambda t, s, g: "totally unrelated words here now")
    text = "The big effect is clear in the data and holds across every sample we tested."
    res = engine.paraphrase(text, engine.Options(mode="balanced", seed=1))
    assert "big effect" in res.text
    assert any("kept as written" in w for w in res.warnings)


def test_service_down_degrades_gracefully(monkeypatch):
    def boom(*a):
        raise engine.TranslationError("down")

    monkeypatch.setattr(engine, "_hop", boom)
    text = "\n".join(f"Paragraph {i} has enough words to be translated properly today." for i in range(8))
    res = engine.paraphrase(text, engine.Options(mode="strong", seed=2))
    assert res.text.count("Paragraph") == 8
    assert any("could not be translated" in w for w in res.warnings)


def test_paragraphs_and_bullets_preserved(monkeypatch):
    monkeypatch.setattr(engine, "_hop", fake_hop)
    text = "Intro\n\n- The big result shows a clear pattern here.\n- Another big claim stands in the text."
    res = engine.paraphrase(text, engine.Options(mode="balanced", seed=3))
    lines = res.text.split("\n")
    assert lines[0] == "Intro" and lines[1] == "" and lines[2].startswith("- ") and lines[3].startswith("- ")


def test_input_validation():
    with pytest.raises(ValueError):
        engine.paraphrase("   ")
    with pytest.raises(ValueError):
        engine.paraphrase("x" * (engine.MAX_INPUT_CHARS + 1))


def test_diff_html_is_escaped():
    out = engine.diff_html("a <script> b", "a <b>x</b> b")
    assert "<script>" not in out and "&lt;" in out
