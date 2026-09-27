"""Text Cleaning Utilities for Business Names and Addresses.

Implements shared base name cleaning and address normalization per Phase 0 specs.
Preserves legal-form suffixes for retrieval IDF.
"""

import re
import unicodedata
from typing import Set

# Compiled regex patterns for name cleaning
RE_BRACKETS = re.compile(r"[\[\]\(\)\{\}<>]")
RE_URL_PREFIX = re.compile(r"https?://\S+", re.IGNORECASE)
RE_EMBEDDED_URL = re.compile(r"\s*\|\s*(?:https?://)?(?:www\.)?\S+", re.IGNORECASE)
RE_WWW = re.compile(r"\bwww\.\S+", re.IGNORECASE)
RE_DOMAIN_SUFFIX = re.compile(r"\.(?:com|org|net|in|co|io)\b", re.IGNORECASE)
RE_LEAD_PUNCT = re.compile(r"^[^\w]+")
RE_SPACES = re.compile(r"\s+")

# Compiled regex patterns for address cleaning
RE_NULL_NA_LITERAL = re.compile(r"\b(?:null|n/a|none|nan)\b|<null>|<n/a>", re.IGNORECASE)
RE_HOUSE_HASH = re.compile(r"#+\s*")
RE_COMMAS = re.compile(r"\s*,\s*")
RE_MULTI_COMMAS = re.compile(r"(?:,\s*)+,")

# Null literal values for has_address evaluation
_NULL_STRINGS: Set[str] = {
    "",
    "null",
    "<null>",
    "n/a",
    "<n/a>",
    "none",
    "nan",
    "undefined",
}


def check_has_address(raw_addr: str | None) -> bool:
    """Evaluate whether the raw address had real content prior to cleaning.

    Treats null/empty, whitespace-only, and literal 'NULL'/'N/A' strings as absent.
    """
    if raw_addr is None:
        return False
    s = str(raw_addr).strip()
    if not s or s.lower() in _NULL_STRINGS:
        return False
    return True


def clean_name_base(raw_name: str | None) -> str:
    """Shared base cleaning pass for business names.

    Performs:
    1. Unicode NFKC normalization and case-folding.
    2. Embedded URL stripping (e.g. '| www.domain.com').
    3. Direct URL / domain suffix stripping (e.g. '.com', 'www.').
    4. Bracket / parenthesis removal.
    5. Leading punctuation run stripping.
    6. Consecutive duplicate token deduplication.
    7. Whitespace collapse.

    Legal suffixes (Ltd/Pvt/Inc/LLC/etc.) are deliberately preserved.
    Invariant: Output is never empty (falls back to stripped raw_name).
    """
    if not raw_name:
        return ""

    raw_str = str(raw_name)
    text = unicodedata.normalize("NFKC", raw_str).lower()

    # Strip embedded URLs like '| www.company.com'
    text = RE_EMBEDDED_URL.sub("", text)
    # Strip protocol and leading www from URLs
    text = re.sub(r"https?://(?:www\.)?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\bwww\.", "", text, flags=re.IGNORECASE)
    # Strip domain suffixes like '.com'
    text = RE_DOMAIN_SUFFIX.sub("", text)

    # Strip bracket characters
    text = RE_BRACKETS.sub(" ", text)

    # Strip leading punctuation runs
    text = RE_LEAD_PUNCT.sub("", text)

    # Deduplicate consecutive identical words
    tokens = text.split()
    if tokens:
        deduped = [tokens[0]]
        for t in tokens[1:]:
            if t != deduped[-1]:
                deduped.append(t)
        text = " ".join(deduped)
    else:
        text = ""

    text = RE_SPACES.sub(" ", text).strip()

    # Guard invariant: name_for_faiss / name_for_bm25 must never be empty
    if not text:
        text = raw_str.strip() or "unknown"

    return text


def clean_address(raw_addr: str | None, has_addr: bool) -> str:
    """Normalize address for addr_for_bm25.

    Performs:
    1. Returns empty string if has_addr is False.
    2. NFKC normalization and lowercase.
    3. Removes placeholder NULL/NA literals embedded in address strings.
    4. Strips leading '#'/ '##' house-number prefixes.
    5. Cleans resulting double commas and collapses whitespace.
    """
    if not has_addr or not raw_addr:
        return ""

    text = unicodedata.normalize("NFKC", str(raw_addr)).lower()

    # Remove placeholder NULL / NA tokens
    text = RE_NULL_NA_LITERAL.sub("", text)

    # Strip '#' or '##' prefixes
    text = RE_HOUSE_HASH.sub("", text)

    # Clean multiple commas
    text = RE_COMMAS.sub(", ", text)
    text = RE_MULTI_COMMAS.sub(",", text)
    text = text.strip(", ")

    # Collapse whitespace
    text = RE_SPACES.sub(" ", text).strip()
    return text
