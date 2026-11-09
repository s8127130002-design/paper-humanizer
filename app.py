"""Streamlit front end for the paraphraser. Logic lives in engine.py."""
import streamlit as st

import engine

st.set_page_config(page_title="Text Paraphraser", layout="wide", initial_sidebar_state="expanded")

st.markdown(
    """
<style>
    .main-title { text-align: center; font-size: 2.3rem; font-weight: 700; margin-bottom: .2rem; }
    .subtitle { text-align: center; opacity: .7; margin-bottom: 1.2rem; }
    .diff-box { border: 1px solid rgba(128,128,128,.35); border-radius: 8px; padding: 14px; line-height: 1.7; }
    .diff-box del { background: rgba(255,80,80,.25); text-decoration: line-through; }
    .diff-box ins { background: rgba(60,180,100,.28); text-decoration: none; }
</style>
""",
    unsafe_allow_html=True,  # static CSS only; user text is never rendered as HTML except escaped diff output
)

MODE_LABELS = {
    "Light": ("light", "No translation. Wording, punctuation and tone cleanup only."),
    "Balanced": ("balanced", "One pivot language (English → X → English). Best meaning retention."),
    "Strong": ("strong", "Two pivot languages. More rewording, more risk of drift."),
}

if "result" not in st.session_state:
    st.session_state.result = None
    st.session_state.run_id = 0

# ------------------------------ Sidebar ------------------------------------ #
with st.sidebar:
    st.header("Settings")
    mode_label = st.select_slider("Rewrite strength", options=list(MODE_LABELS), value="Balanced")
    st.caption(MODE_LABELS[mode_label][1])
    tone = st.selectbox(
        "Tone", engine.TONES, index=0,
        help="Academic expands contractions. Conversational adds them. Neutral leaves them as is.",
    )
    st.subheader("Cleanup")
    replace_wording = st.checkbox("Simplify stock wording", value=True)
    fix_dashes = st.checkbox("Replace em dashes", value=True)
    strip_invisible = st.checkbox("Remove hidden characters", value=True,
                                  help="Zero-width spaces, soft hyphens, bidi marks, odd spaces.")
    strip_chatbot = st.checkbox("Remove chatbot intro and sign-off lines", value=True)
    with st.expander("Advanced"):
        pivots = st.multiselect(
            "Pivot languages", options=list(engine.PIVOT_LANGUAGES),
            default=list(engine.PIVOT_LANGUAGES), format_func=lambda c: engine.PIVOT_LANGUAGES[c],
        )
        seed = st.number_input("Seed (0 = random)", min_value=0, value=0, step=1,
                               help="Use the same seed to repeat the same language path.")
    st.divider()
    if st.button("Clear"):
        st.session_state.result = None
        st.rerun()

# ------------------------------ Main --------------------------------------- #
st.markdown('<p class="main-title">Text Paraphraser</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Back-translation plus wording cleanup. URLs, citations and numbers are protected.</p>',
    unsafe_allow_html=True,
)

with st.form("input_form"):
    source = st.text_area(
        "Input text", height=260, max_chars=engine.MAX_INPUT_CHARS,
        placeholder="Paste the text you want to paraphrase here...", label_visibility="collapsed",
    )
    submitted = st.form_submit_button("Paraphrase", type="primary")

if submitted:
    if not source.strip():
        st.warning("Please paste some text first.")
    else:
        opts = engine.Options(
            mode=MODE_LABELS[mode_label][0], tone=tone, strip_invisible=strip_invisible,
            strip_chatbot=strip_chatbot, fix_dashes=fix_dashes, replace_wording=replace_wording,
            pivots=tuple(pivots), seed=int(seed) or None,
        )
        bar = st.progress(0.0, text="Starting...")
        try:
            st.session_state.result = engine.paraphrase(
                source, opts,
                progress=lambda f: bar.progress(min(f, 1.0), text=f"Paraphrasing... {int(f * 100)}%"),
            )
            st.session_state.run_id += 1
        except ValueError as exc:
            st.error(str(exc))
        except Exception as exc:  # last-resort guard so the app never shows a raw traceback
            st.error("Something went wrong while paraphrasing.")
            with st.expander("Details"):
                st.exception(exc)
        finally:
            bar.empty()

st.divider()
st.subheader("Output")

res = st.session_state.result
if res is None:
    st.info("Your paraphrased text will appear here.")
else:
    for w in res.warnings:
        st.warning(w)

    # Plain text widget: user text is never injected as HTML, and it is selectable/copyable.
    st.text_area("Paraphrased text", value=res.text, height=260,
                 key=f"output_{st.session_state.run_id}", label_visibility="collapsed")

    s = res.stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Words in → out", f"{s['words_in']} → {s['words_out']}")
    c2.metric("Wording changed", f"{s['changed_pct']}%")
    c3.metric("Time", f"{s['seconds']} s")
    c4.metric("Path", s["path"])

    st.download_button("Download as .txt", res.text, file_name="paraphrased_text.txt", mime="text/plain")

    with st.expander("Show changes"):
        diff = engine.diff_html(res.source, res.text)
        if diff is None:
            st.caption("Text is too long for an inline diff.")
        else:
            st.markdown(f'<div class="diff-box">{diff}</div>', unsafe_allow_html=True)
    with st.expander("Copy view"):
        st.code(res.text, language=None)

st.caption(
    "Always read the result against your original. Back-translation can shift meaning, "
    "and no tool can guarantee how a given AI detector will score a text."
)
