# Text Paraphraser

A Streamlit app that paraphrases text with back-translation (English → pivot language → English) followed by a conservative wording cleanup.

## Features
- Three strengths: Light (rules only), Balanced (1 pivot language), Strong (2 pivot languages)
- URLs, emails, DOIs, citations like (Smith, 2020), `[1]` references, numbers and percentages are protected from translation
- Long text is split into sentence-aligned chunks and translated in parallel with retries
- Per-chunk quality guard: if a round trip loses a protected item or drifts too far in meaning, that chunk is kept as written and you get a warning
- Tone control: academic (no contractions), neutral, conversational
- Hidden-character removal, em dash replacement, chatbot intro/sign-off removal
- Word-level "Show changes" view, plain-text output (no HTML injection), .txt download

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Test
```bash
pip install pytest
pytest -q
```

## Deploy
Push to GitHub, then create the app at share.streamlit.io with `app.py` as the entry point.

## Notes
- Translation uses the free Google Translate endpoint through `deep-translator`. It can rate-limit heavy use; the app retries and reports blocks it could not translate.
- Input limit is 20,000 characters per run.
- Always compare the output with your original. No tool can guarantee how a particular AI detector will score a text.
