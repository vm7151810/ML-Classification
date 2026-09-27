"""Script-Family Transliteration Registry & Dispatcher.

Dispatches cross-script names to appropriate script-family transliteration handlers.
For Indic scripts, maps to ASCII using indic-transliteration (OPTITRANS scheme).
Unhandled script families pass through native script untouched and are logged.
"""

import re
import unicodedata
from typing import Tuple, Dict
from indic_transliteration import sanscript

# Re-normalization pattern for transliterated text
_RE_SPACES = re.compile(r"\s+")

# Supported Indic script names mapped to sanscript scheme identifiers
_INDIC_SCRIPTS: Dict[str, str] = {
    "Devanagari": sanscript.DEVANAGARI,
    "Tamil": sanscript.TAMIL,
    "Telugu": sanscript.TELUGU,
    "Kannada": sanscript.KANNADA,
    "Bengali": sanscript.BENGALI,
    "Gujarati": sanscript.GUJARATI,
    "Gurmukhi": sanscript.GURMUKHI,
    "Malayalam": sanscript.MALAYALAM,
    "Oriya": sanscript.ORIYA,
}

# Pre-transliteration normalizations for unmapped/chillus/consonants in Indic scripts
_INDIC_PRE_TRANS: Dict[str, str] = {
    # Tamil NNNA (ன, U+0BA9) -> Dental NA (ந, U+0BA8) which transliterates cleanly to 'n'
    "\u0ba9": "\u0ba8",
    # Malayalam Chillu letters -> Base consonant + Virama
    "\u0d7a": "ണ്",  # ൺ -> ണ + ്
    "\u0d7b": "ന്",  # ൻ -> ന + ്
    "\u0d7c": "ര്",  # ർ -> ര + ്
    "\u0d7d": "ല്",  # ൽ -> ല + ്
    "\u0d7e": "ള്",  # ൾ -> ള + ്
    "\u0d7f": "ക്",  # ൿ -> ക + ്
    # Malayalam AU Length Mark -> AU vowel sign
    "\u0d57": "ൌ",  # ൗ -> ൌ
    # Oriya WA (ୱ, U+0B71) -> BA (ବ, U+0B2C)
    "\u0b71": "\u0b2c",
}
_RE_INDIC_PRE = re.compile("|".join(re.escape(k) for k in _INDIC_PRE_TRANS.keys()))

# Post-transliteration cleanup: strip zero-width joiners and lone nukta combining marks
_RE_STRIP_MARKS = re.compile(r"[\u200c\u200d\ufeff\u093c\u09bc\u0a3c\u0abc\u0b3c\u0cbc\u0d3c\u0bbc]")


def _clean_transliterated_text(text: str) -> str:
    """Re-normalize transliterated text: strip marks, NFKC, lowercase, collapse whitespace."""
    if not text:
        return ""
    # Strip invisible zero-width formatting characters and lone nukta diacritics
    text = _RE_STRIP_MARKS.sub("", text)
    text = unicodedata.normalize("NFKC", text).lower()
    text = _RE_SPACES.sub(" ", text).strip()
    return text


def transliterate_name(
    cleaned_name: str,
    script: str,
    is_cross_script: bool,
) -> Tuple[str, bool, bool]:
    """Generate name_for_bm25 via script-family dispatch.

    Args:
        cleaned_name: Cleaned native-script business name (shared base pass).
        script: Dominant script detected for the business name.
        is_cross_script: Whether the dominant script is non-Latin.

    Returns:
        (name_for_bm25, was_handled, is_degenerate)
        - name_for_bm25: Transliterated name if handled, else native cleaned_name.
        - was_handled: True if an explicit transliteration handler processed the row.
        - is_degenerate: True if transliteration output was empty or has surviving non-ASCII codepoints.
    """
    # Latin or non-cross-script rows pass through untouched
    if not is_cross_script:
        return cleaned_name, False, False

    # Check for Indic-family script handler
    scheme = _INDIC_SCRIPTS.get(script)
    if scheme is not None:
        try:
            # Apply pre-transliteration character mappings
            preprocessed = _RE_INDIC_PRE.sub(lambda m: _INDIC_PRE_TRANS[m.group(0)], cleaned_name)
            translit = sanscript.transliterate(preprocessed, scheme, sanscript.OPTITRANS)
            normalized = _clean_transliterated_text(translit)

            if not normalized:
                # Degenerate case: output was empty, fallback to native cleaned name
                return cleaned_name, True, True

            # Degenerate check: flag if any native/non-ASCII character survived untransliterated
            has_surviving_non_ascii = any(ord(c) > 127 for c in normalized)
            if has_surviving_non_ascii:
                return normalized, True, True

            return normalized, True, False
        except Exception:
            return cleaned_name, False, True

    # Unhandled script family (e.g. Arabic, Cyrillic, CJK, etc.)
    # Deliberate logged gap: pass native script through unchanged
    return cleaned_name, False, False
