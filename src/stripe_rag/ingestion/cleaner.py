"""
Markdown cleaning step applied between crawl output and chunking.

Issues addressed (discovered by inspecting 5 sample pages):
1. Repeated JSON response blobs on API reference pages (truncate large code blocks)
2. Language-selector tab UI with header ("Select a language / Ruby / Python / No results")
3. Standalone language-list blocks without header (Ruby / Python / PHP / Java / ...)
4. Standalone code-block labels ("Command Line", "Response", "Gemfile", etc.)
5. Concatenated country/region tab-selector lines (e.g. "AustraliaAustriaBelgium...")
6. Field name + type annotation merging ("codenullable string" → "code (nullable string)")
"""
from __future__ import annotations

import re

# ── 1. Code-block truncation ──────────────────────────────────────────────────

_CODE_BLOCK_RE = re.compile(r"(```[^\n]*\n)(.*?)(```)", re.DOTALL)
_MAX_CODE_LINES = 50
_KEEP_CODE_LINES = 20


def _truncate_code_blocks(text: str) -> str:
    def repl(m: re.Match) -> str:
        fence_open, body, fence_close = m.group(1), m.group(2), m.group(3)
        lines = body.splitlines()
        if len(lines) > _MAX_CODE_LINES:
            kept = "\n".join(lines[:_KEEP_CODE_LINES])
            return f"{fence_open}{kept}\n# ... (truncated)\n{fence_close}"
        return m.group(0)
    return _CODE_BLOCK_RE.sub(repl, text)


# ── 2. Language-selector UI with "Select a language" header ──────────────────

_LANG_NAMES = (
    "Ruby", "Python", "PHP", "Java", "Node\\.js", "Go", r"\.NET",
    "Stripe CLI", "cURL", "Rails", "Django",
)
_LANG_NAMES_PAT = "|".join(_LANG_NAMES)

_LANG_SELECTOR_RE = re.compile(
    r"(?:Command Line\n\n)?"
    r"Select a language\n\n"
    r"(?:(?:" + _LANG_NAMES_PAT + r")\n\n)*"
    r"No results\n?",
    re.MULTILINE,
)

# ── 3. Standalone language-list blocks (no "Select a language" header) ────────
# Matches a block that consists *only* of known language names as individual
# paragraphs, e.g.:  Ruby\n\nPython\n\nPHP\n\nJava\n\nNode.js\n\nGo\n\n.NET
# The block must contain at least 3 distinct language names to fire.

_STANDALONE_LANG_BLOCK_RE = re.compile(
    r"(?:(?:" + _LANG_NAMES_PAT + r")\n\n){3,}"
    r"(?:" + _LANG_NAMES_PAT + r")\n?",
    re.MULTILINE,
)

# ── 4. Standalone orphaned UI labels ─────────────────────────────────────────

_STANDALONE_LABELS = [
    "Command Line",
    "Gemfile",
    "Response",
    "Authenticated Request",
    "Your API Key",
    "Client Libraries",
    "Base URL",
    "No results",
    "Select a language",
]
_STANDALONE_LABELS_RE = re.compile(
    r"^(" + "|".join(re.escape(l) for l in _STANDALONE_LABELS) + r")\n",
    re.MULTILINE,
)

# ── 5. Concatenated country/region tab-selector lines ─────────────────────────

_COUNTRY_LINE_RE = re.compile(r"^[A-Z][A-Za-z ]{79,}$", re.MULTILINE)


def _is_country_blob(line: str) -> bool:
    """Space density < 8 % ⟹ likely concatenated tokens, not a sentence."""
    return len(line) >= 80 and (line.count(" ") / len(line)) < 0.08


def _strip_country_blobs(text: str) -> str:
    def repl(m: re.Match) -> str:
        return "" if _is_country_blob(m.group(0)) else m.group(0)
    return _COUNTRY_LINE_RE.sub(repl, text)


# ── 6. Field name + type annotation merging (API reference pages) ─────────────
#
# The HTML-to-markdown conversion produces:
#   "codenullable string"  /  "typeenum"  /  "idstring"
# because the type badge was inline text with no separator.
#
# We rewrite these to "code (nullable string)" / "type (enum)" / "id (string)"
# ONLY outside code fences to avoid mangling actual code.
#
# Type vocabulary (from Stripe API docs):
#   nullable <base>  |  string  |  object  |  integer  |  boolean
#   enum  |  timestamp  |  array  |  number  |  hash  |  float

_BASE_TYPES = (
    "string", "object", "integer", "boolean", "enum",
    "timestamp", "array", "number", "hash", "float",
)
_TYPE_PAT = "nullable (?:" + "|".join(_BASE_TYPES) + r")|\b" + r"\b|\b".join(_BASE_TYPES) + r"\b"

# Match a snake_case / camelCase identifier immediately followed by a type token
# e.g.  "codenullable string"  "typeenum"  "payment_intentnullable object"
_FIELD_TYPE_RE = re.compile(
    r"\b([\w_]+?)(" + _TYPE_PAT + r")(?=\s|$|\n|\)|,)",
)

# Extra badge text that appears after the type on the same token run
_BADGE_SUFFIXES_RE = re.compile(
    r"\(nullable \w+\)(Preview feature|Expandable|Required(?: conditionally)?|US only)?",
)


def _fix_field_types_in_text(text: str) -> str:
    """Reformat merged field+type tokens → 'fieldname (type)'."""
    return _FIELD_TYPE_RE.sub(r"\1 (\2)", text)


def _fix_field_types(full_text: str) -> str:
    """Apply _fix_field_types_in_text only outside code fences."""
    parts: list[str] = []
    last = 0
    for m in _CODE_BLOCK_RE.finditer(full_text):
        # Text before the code block — apply fix
        parts.append(_fix_field_types_in_text(full_text[last:m.start()]))
        # Code block itself — keep verbatim
        parts.append(m.group(0))
        last = m.end()
    # Remaining text after last code block
    parts.append(_fix_field_types_in_text(full_text[last:]))
    return "".join(parts)


# ── Public API ────────────────────────────────────────────────────────────────

def clean_markdown(text: str) -> str:
    """Apply all cleaning passes and return sanitised markdown."""
    # Pass 1: truncate large code blocks (do first — protects code from later passes)
    text = _truncate_code_blocks(text)
    # Pass 2: language-selector UI with header
    text = _LANG_SELECTOR_RE.sub("", text)
    # Pass 3: standalone language-list blocks (no header)
    text = _STANDALONE_LANG_BLOCK_RE.sub("", text)
    # Pass 4: orphaned single-line UI labels
    text = _STANDALONE_LABELS_RE.sub("", text)
    # Pass 5: concatenated country/region blobs
    text = _strip_country_blobs(text)
    # Pass 6: field name + type annotation merging
    text = _fix_field_types(text)
    # Pass 7: normalise excessive blank lines left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()
