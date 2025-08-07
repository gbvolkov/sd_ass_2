# --- Install once (uncomment in your environment) ---
# !pip install -U spacy scispacy
# !python -m spacy download ru_core_news_sm

import re
import spacy
from scispacy.abbreviation import AbbreviationDetector

def build_ru_nlp() -> spacy.language.Language:
    """Russian spaCy pipeline + SciSpaCy AbbreviationDetector (CPU-friendly)."""
    nlp = spacy.load("ru_core_news_sm", disable=["ner", "textcat"])
    # Ensure sentence boundaries are available
    if "parser" not in nlp.pipe_names and "senter" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")
    # Add abbreviation detector (Schwartz–Hearst algorithm)
    if "abbreviation_detector" not in nlp.pipe_names:
        nlp.add_pipe("abbreviation_detector")
    return nlp

# Regexes for standalone abbreviations (not necessarily defined inline)
RE_ALLCAPS = re.compile(r"^[A-ZА-ЯЁ]{2,10}$", flags=re.UNICODE)
# Dotted abbreviations: e.g., "т.е.", "т.к.", "и т.д." (compact, no spaces)
RE_DOTTED = re.compile(r"^(?:[A-Za-zА-Яа-яЁё]\.){2,}$", flags=re.UNICODE)

# Optional: a small set of roman numerals to filter out common false positives
ROMAN_NUMERALS = {
    "I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII","XIII","XIV","XV"
}

def extract_abbreviations_ru(
    text: str,
    nlp: spacy.language.Language,
    include_standalone: bool = True
):
    """
    Returns:
      {
        'pairs': [{'short': 'СОКР', 'long': 'Длинная Форма', 'short_span': (s,e), 'long_span': (s,e)}, ...],
        'standalone': ['РФ','ООО','ЕАЭС','т.е.', ...]   # optional
      }
    """
    doc = nlp(text)

    # 1) Abbreviation–definition pairs from the Schwartz–Hearst algorithm
    pairs = []
    for abrv in doc._.abbreviations:
        long_form = abrv._.long_form
        pairs.append({
            "short": abrv.text,
            "long": long_form.text,
            "short_span": (abrv.start_char, abrv.end_char),
            "long_span": (long_form.start_char, long_form.end_char),
        })

    # Deduplicate (keep one per (short,long))
    def _key(p): return (p["short"].strip().upper(), p["long"].strip().lower())
    pairs = list({ _key(p): p for p in pairs }.values())

    result = {"pairs": pairs}

    # 2) Optional: find standalone abbreviations (not defined inline)
    if include_standalone:
        defined_shorts = {p["short"] for p in pairs}

        # (a) token-level all-caps acronyms (Cyrillic or Latin)
        candidates = set()
        for tok in doc:
            t = tok.text
            # skip tokens with digits or hyphens to reduce noise; adjust if needed
            if any(ch.isdigit() for ch in t) or "-" in t:
                continue
            if t.isalpha() and RE_ALLCAPS.match(t) and t not in ROMAN_NUMERALS:
                if t not in defined_shorts:
                    candidates.add(t)

        # (b) dotted abbreviations present in the raw text (may be single token or split)
        for m in RE_DOTTED.finditer(text):
            dotted = m.group(0)
            if dotted not in defined_shorts:
                candidates.add(dotted)

        result["standalone"] = sorted(candidates)

    return result


# ---------------- Example ----------------
if __name__ == "__main__":
    nlp = build_ru_nlp()
    sample = """
    Настоящее Положение соответствует Общему регламенту по защите данных (GDPR)
    и законодательству РФ. Оценка воздействия на защиту данных (DPIA) проводится при необходимости.
    Европейское химическое агентство (ECHA) публикует руководства, применимые в ЕАЭС.
    В тексте встречаются сокращения, например, т.е. и т.п., а также упоминания РФ и ООО.
    """
    out = extract_abbreviations_ru(sample, nlp)
    from pprint import pprint
    pprint(out)