"""Cross-lingual leakage experiment (Part 1 of the proposal).

Erase a concept with an edit tuned in English, then ask the concept's held-out
MC questions in en/de/es/fr/ru and compare against the unedited model in the
same language. Modules:

- ``data``         -- load the translated MC items, language-aligned shuffle
- ``prompts``      -- MC prompt scaffold in each language (native wrapper)
- ``mc_eval``      -- evaluate one (concept, language) cell, return records
- ``run_baseline`` -- CLI: pristine-model baseline over concepts x languages

The Gemini-era implementation lives in ``archive_gemini/crosslingual/``.
"""
