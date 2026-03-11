"""
Heading-aware recursive chunker.

Algorithm:
  1. Split page markdown on h1-h4 boundaries → (heading_path, text) sections
  2. For each section:
       - token_count ≤ MAX_TOKENS  → single chunk
       - token_count  > MAX_TOKENS → recursive split: \\n\\n → \\n → sentence → char
         with OVERLAP_TOKENS sliding window
  3. Prepend heading_path to every chunk's content
  4. Discard chunks with token_count < MIN_CHUNK_TOKENS
"""
from __future__ import annotations

import re
import uuid
from dataclasses import dataclass

import tiktoken

# ── Constants ─────────────────────────────────────────────────────────────────

MAX_TOKENS = 512
OVERLAP_TOKENS = 64
MIN_CHUNK_TOKENS = 50
_ENCODER = tiktoken.get_encoding("cl100k_base")

# ── Heading regex: matches lines like "# Foo", "## Bar", etc. ─────────────────
_HEADING_RE = re.compile(r"^(#{1,4})\s+(.+)$", re.MULTILINE)


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class Chunk:
    chunk_id: str
    source_url: str
    page_title: str
    section_prefix: str
    heading_path: str
    content: str          # heading_path prepended
    token_count: int
    char_count: int
    chunk_index: int
    total_chunks: int     # filled in after all chunks for a page are known

    def to_dict(self) -> dict:
        return {
            "chunk_id": self.chunk_id,
            "source_url": self.source_url,
            "page_title": self.page_title,
            "section_prefix": self.section_prefix,
            "heading_path": self.heading_path,
            "content": self.content,
            "token_count": self.token_count,
            "char_count": self.char_count,
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


# ── Token helpers ─────────────────────────────────────────────────────────────

def _count_tokens(text: str) -> int:
    return len(_ENCODER.encode(text, disallowed_special=()))


def _decode_tokens(tokens: list[int]) -> str:
    return _ENCODER.decode(tokens)


# ── Stage 1: heading split ────────────────────────────────────────────────────

def _split_by_headings(markdown: str) -> list[tuple[list[str], str]]:
    """
    Split markdown into (heading_stack, body_text) pairs.
    heading_stack is a list of heading strings at decreasing depth,
    e.g. ["API Reference", "Errors", "Attributes"].
    """
    sections: list[tuple[list[str], str]] = []
    stack: list[tuple[int, str]] = []  # (level, text)

    # Find all heading positions
    matches = list(_HEADING_RE.finditer(markdown))

    # Text before first heading (page preamble)
    preamble_end = matches[0].start() if matches else len(markdown)
    preamble = markdown[:preamble_end].strip()

    for i, m in enumerate(matches):
        level = len(m.group(1))   # number of # characters
        heading_text = m.group(2).strip()

        # Pop stack entries at same level or deeper
        stack = [(l, t) for l, t in stack if l < level]
        stack.append((level, heading_text))

        # Body text: from end of this heading line to start of next heading
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        body = markdown[body_start:body_end].strip()

        heading_stack = [t for _, t in stack]
        sections.append((heading_stack, body))

    # Prepend preamble as a top-level section if it has content
    if preamble:
        first_stack = [sections[0][0][0]] if sections else ["Introduction"]
        sections.insert(0, (first_stack, preamble))

    return sections


# ── Stage 2: recursive split ──────────────────────────────────────────────────

def _split_text_recursively(text: str, max_tokens: int, overlap_tokens: int) -> list[str]:
    """
    Split `text` into pieces each ≤ max_tokens with overlap_tokens overlap.
    Tries progressively finer separators: paragraph → line → sentence → char.
    """
    tokens = _ENCODER.encode(text, disallowed_special=())
    if len(tokens) <= max_tokens:
        return [text] if text.strip() else []

    separators = ["\n\n", "\n", ". ", " ", ""]

    for sep in separators:
        if sep:
            parts = text.split(sep)
        else:
            # Character-level fallback: split into token windows directly
            return _token_windows(tokens, max_tokens, overlap_tokens)

        if len(parts) <= 1:
            continue

        chunks: list[str] = []
        current_tokens: list[int] = []

        for part in parts:
            part_text = part + (sep if sep else "")
            part_tokens = _ENCODER.encode(part_text, disallowed_special=())

            if len(current_tokens) + len(part_tokens) > max_tokens:
                if current_tokens:
                    chunks.append(_decode_tokens(current_tokens).strip())
                # Start new window with overlap; clamp so window never exceeds max_tokens
                current_tokens = (current_tokens[-overlap_tokens:] + part_tokens)[:max_tokens]
            else:
                current_tokens.extend(part_tokens)

        if current_tokens:
            chunks.append(_decode_tokens(current_tokens).strip())

        # If we made progress (more than one chunk), return
        if len(chunks) > 1:
            return [c for c in chunks if c.strip()]

    return [text]


def _token_windows(tokens: list[int], max_tokens: int, overlap_tokens: int) -> list[str]:
    """Last-resort: sliding token windows."""
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + max_tokens, len(tokens))
        chunks.append(_decode_tokens(tokens[start:end]).strip())
        if end == len(tokens):
            break
        start += max_tokens - overlap_tokens
    return [c for c in chunks if c.strip()]


# ── Stage 3: build Chunk objects ─────────────────────────────────────────────

def _make_content(heading_path: str, body: str) -> str:
    """Prepend heading_path to body so the embedding carries section context."""
    return f"{heading_path}\n\n{body}" if body else heading_path


# ── Public API ────────────────────────────────────────────────────────────────

class HeadingAwareChunker:
    def __init__(
        self,
        max_tokens: int = MAX_TOKENS,
        overlap_tokens: int = OVERLAP_TOKENS,
        min_chunk_tokens: int = MIN_CHUNK_TOKENS,
    ) -> None:
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_chunk_tokens = min_chunk_tokens

    def chunk_page(
        self,
        markdown: str,
        source_url: str,
        page_title: str,
        section_prefix: str,
    ) -> list[Chunk]:
        sections = _split_by_headings(markdown)
        raw_chunks: list[tuple[str, str]] = []  # (heading_path, text)

        for heading_stack, body in sections:
            heading_path = " > ".join(heading_stack)

            if not body.strip():
                # Headings with no body — skip
                continue

            content = _make_content(heading_path, body)
            token_count = _count_tokens(content)

            if token_count <= self.max_tokens:
                raw_chunks.append((heading_path, content))
            else:
                # Recursive split on the body, then re-prepend heading
                heading_overhead = _count_tokens(heading_path + "\n\n")
                max_body = max(self.max_tokens - heading_overhead, self.min_chunk_tokens)
                sub_texts = _split_text_recursively(body, max_body, self.overlap_tokens)
                for sub in sub_texts:
                    full = _make_content(heading_path, sub)
                    # Hard fallback: if content is still oversized (e.g. dense table),
                    # force-split on token windows to guarantee the upper bound.
                    if _count_tokens(full) > self.max_tokens:
                        tokens = _ENCODER.encode(full, disallowed_special=())
                        for win in _token_windows(tokens, self.max_tokens, self.overlap_tokens):
                            raw_chunks.append((heading_path, win))
                    else:
                        raw_chunks.append((heading_path, full))

        # Build Chunk objects, filter by min tokens
        chunks: list[Chunk] = []
        for heading_path, content in raw_chunks:
            token_count = _count_tokens(content)
            if token_count < self.min_chunk_tokens:
                continue
            chunks.append(
                Chunk(
                    chunk_id=str(uuid.uuid4()),
                    source_url=source_url,
                    page_title=page_title,
                    section_prefix=section_prefix,
                    heading_path=heading_path,
                    content=content,
                    token_count=token_count,
                    char_count=len(content),
                    chunk_index=len(chunks),
                    total_chunks=0,  # filled below
                )
            )

        # Fill in total_chunks
        total = len(chunks)
        for c in chunks:
            c.total_chunks = total

        return chunks
