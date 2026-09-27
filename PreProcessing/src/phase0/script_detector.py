"""Open-set Unicode Script Detector.

Detects the dominant script in a given business name using Unicode character
properties in an open-set manner, without hardcoding fixed script enums.
"""

from collections import Counter
import unicodedata
from typing import Tuple

# Fast cache for codepoint -> script name mapping
_CHAR_SCRIPT_CACHE = {}


def get_char_script(char: str) -> str | None:
    """Return the script name for a single character via Unicode standard naming."""
    cached = _CHAR_SCRIPT_CACHE.get(char)
    if cached is not None:
        return cached

    cat = unicodedata.category(char)
    # Only consider Letters (L) and Marks (M, e.g. vowel signs, matras, viramas)
    if not (cat.startswith("L") or cat.startswith("M")):
        _CHAR_SCRIPT_CACHE[char] = None
        return None

    try:
        uname = unicodedata.name(char)
        # Unicode names typically begin with the script name, e.g.:
        # 'LATIN SMALL LETTER A', 'DEVANAGARI LETTER RA', 'TAMIL LETTER TA'
        first_token = uname.split()[0].capitalize()
        # Normalise common acronyms / script naming conventions
        if first_token == "Cjk":
            script_name = "CJK"
        else:
            script_name = first_token
    except ValueError:
        script_name = None

    _CHAR_SCRIPT_CACHE[char] = script_name
    return script_name


def detect_script(text: str) -> Tuple[str, bool]:
    """Detect dominant script and cross-script flag for a business name.

    Returns:
        (dominant_script, is_cross_script)
        - dominant_script: Detected script name (e.g. 'Latin', 'Devanagari', 'Tamil', etc.)
        - is_cross_script: True if business_name is not predominantly Latin script.
    """
    if not text:
        return "Latin", False

    # Fast path for pure ASCII text (covers ~90%+ of standard datasets)
    if text.isascii():
        return "Latin", False

    counts = Counter()
    for ch in text:
        s = get_char_script(ch)
        if s:
            counts[s] += 1

    if not counts:
        # No alphabetic or mark characters found; default to Latin
        return "Latin", False

    # Check if any non-Latin letters/marks exist
    non_latin_counts = {s: c for s, c in counts.items() if s != "Latin"}
    if non_latin_counts:
        # Business name contains cross-script text that requires transliteration
        is_cross_script = True
        # Select the dominant non-Latin script to guide the transliteration scheme
        dominant_script = max(non_latin_counts.items(), key=lambda x: x[1])[0]
    else:
        is_cross_script = False
        dominant_script = "Latin"

    return dominant_script, is_cross_script
