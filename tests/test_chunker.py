"""Phase 3 unit tests — HeadingAwareChunker."""
from __future__ import annotations

import pytest

from stripe_rag.ingestion.chunker import (
    MAX_TOKENS,
    MIN_CHUNK_TOKENS,
    HeadingAwareChunker,
    _count_tokens,
    _split_by_headings,
    _split_text_recursively,
)

# ── Fixtures ──────────────────────────────────────────────────────────────────

SIMPLE_MD = """\
# Charges

A Charge object represents a payment attempt.

## Create a charge

To create a charge send a POST request to the charges endpoint.

### Parameters

- **amount** (required): A positive integer representing the charge amount.
- **currency** (required): Three-letter ISO currency code.
- **source** (required): A payment source to charge.

### Returns

Returns a Charge object if the call succeeds.

## List charges

Returns a list of charges previously created.

### Parameters

- **limit**: A limit on the number of objects to be returned (1–100).
- **starting_after**: A cursor for pagination.
"""

DEEP_HEADINGS_MD = """\
# Top

## Level 2a

Content under 2a.

### Level 3a

Content under 3a.

#### Level 4

Very deep content.

### Level 3b

Back to level 3.

## Level 2b

Content under 2b — 2a stack cleared.
"""

# ~600-token body to force recursive splitting
LONG_SECTION_BODY = " ".join(["This is a detailed explanation of the Stripe API."] * 60)
LONG_MD = f"""\
# Long Page

## Dense Section

{LONG_SECTION_BODY}
"""


# ── _split_by_headings ────────────────────────────────────────────────────────

def test_split_basic_structure():
    sections = _split_by_headings(SIMPLE_MD)
    heading_paths = [" > ".join(hs) for hs, _ in sections]
    assert any("Charges" in p for p in heading_paths)
    assert any("Create a charge" in p for p in heading_paths)
    assert any("Parameters" in p for p in heading_paths)


def test_split_heading_stack_resets_on_same_level():
    sections = _split_by_headings(SIMPLE_MD)
    # "List charges" should NOT contain "Create a charge" in its stack
    for hs, _ in sections:
        path = " > ".join(hs)
        if "List charges" in path:
            assert "Create a charge" not in path


def test_split_deep_headings_stack():
    sections = _split_by_headings(DEEP_HEADINGS_MD)
    paths = [" > ".join(hs) for hs, _ in sections]

    # Level 4 should have full ancestry
    l4_paths = [p for p in paths if "Level 4" in p]
    assert l4_paths, "Level 4 section not found"
    assert "Top" in l4_paths[0]
    assert "Level 2a" in l4_paths[0]
    assert "Level 3a" in l4_paths[0]

    # Level 2b should NOT have Level 2a in its stack
    l2b_paths = [p for p in paths if "Level 2b" in p]
    assert l2b_paths
    assert "Level 2a" not in l2b_paths[0]


def test_split_level3b_doesnt_contain_level3a():
    sections = _split_by_headings(DEEP_HEADINGS_MD)
    for hs, _ in sections:
        if "Level 3b" in hs:
            assert "Level 3a" not in hs


# ── HeadingAwareChunker.chunk_page ────────────────────────────────────────────

def make_chunker(**kwargs) -> HeadingAwareChunker:
    return HeadingAwareChunker(**kwargs)


def test_chunk_page_returns_chunks():
    chunker = make_chunker()
    chunks = chunker.chunk_page(SIMPLE_MD, "https://docs.stripe.com/api/charges", "Charges", "api")
    assert len(chunks) > 0


def test_chunk_token_bounds():
    chunker = make_chunker()
    chunks = chunker.chunk_page(SIMPLE_MD, "https://example.com", "Test", "api")
    for c in chunks:
        assert c.token_count >= MIN_CHUNK_TOKENS, f"Chunk too small: {c.token_count}"
        assert c.token_count <= MAX_TOKENS, f"Chunk too large: {c.token_count}"


def test_chunk_heading_path_nonempty():
    chunker = make_chunker()
    chunks = chunker.chunk_page(SIMPLE_MD, "https://example.com", "Test", "api")
    for c in chunks:
        assert c.heading_path, "heading_path must not be empty"


def test_chunk_content_contains_heading_path():
    chunker = make_chunker()
    chunks = chunker.chunk_page(SIMPLE_MD, "https://example.com", "Test", "api")
    for c in chunks:
        assert c.heading_path in c.content, "content must be prefixed with heading_path"


def test_chunk_metadata_fields():
    chunker = make_chunker()
    url = "https://docs.stripe.com/api/charges"
    chunks = chunker.chunk_page(SIMPLE_MD, url, "Charges", "api")
    for c in chunks:
        assert c.source_url == url
        assert c.page_title == "Charges"
        assert c.section_prefix == "api"
        assert c.chunk_id  # non-empty UUID
        assert c.char_count == len(c.content)


def test_chunk_index_and_total():
    chunker = make_chunker()
    chunks = chunker.chunk_page(SIMPLE_MD, "https://example.com", "Test", "api")
    total = len(chunks)
    for i, c in enumerate(chunks):
        assert c.chunk_index == i
        assert c.total_chunks == total


def test_chunk_to_dict_keys():
    chunker = make_chunker()
    chunks = chunker.chunk_page(SIMPLE_MD, "https://example.com", "Test", "api")
    required = {
        "chunk_id", "source_url", "page_title", "section_prefix",
        "heading_path", "content", "token_count", "char_count",
        "chunk_index", "total_chunks",
    }
    for c in chunks:
        assert required == set(c.to_dict().keys())


def test_long_section_is_split():
    """A section exceeding MAX_TOKENS must produce multiple chunks."""
    chunker = make_chunker()
    chunks = chunker.chunk_page(LONG_MD, "https://example.com", "Long", "api")
    dense_chunks = [c for c in chunks if "Dense Section" in c.heading_path]
    assert len(dense_chunks) > 1, "Dense section should have been split"


def test_long_section_all_chunks_within_bounds():
    chunker = make_chunker()
    chunks = chunker.chunk_page(LONG_MD, "https://example.com", "Long", "api")
    for c in chunks:
        assert c.token_count <= MAX_TOKENS


def test_small_section_discarded():
    """Sections smaller than MIN_CHUNK_TOKENS must be dropped."""
    md = "# Title\n\n## Tiny\n\nHi.\n\n## Real Section\n\n" + ("word " * 60)
    chunker = make_chunker()
    chunks = chunker.chunk_page(md, "https://example.com", "T", "api")
    # "Tiny" section body is 1 word — should be filtered
    tiny_chunks = [c for c in chunks if "Tiny" in c.heading_path and "Real Section" not in c.heading_path]
    assert len(tiny_chunks) == 0, "Tiny section should have been discarded"


def test_empty_markdown_returns_no_chunks():
    chunker = make_chunker()
    chunks = chunker.chunk_page("", "https://example.com", "T", "api")
    assert chunks == []


def test_no_headings_page():
    """A page with no headings should still produce chunks from preamble."""
    md = "This is a page with no headings. " * 20
    chunker = make_chunker()
    chunks = chunker.chunk_page(md, "https://example.com", "NoHeadings", "payments")
    assert len(chunks) >= 1


# ── _split_text_recursively ───────────────────────────────────────────────────

def test_recursive_split_short_text_unchanged():
    text = "Short text that fits in one chunk."
    result = _split_text_recursively(text, max_tokens=512, overlap_tokens=64)
    assert result == [text]


def test_recursive_split_long_text():
    long_text = "This is a sentence. " * 60
    result = _split_text_recursively(long_text, max_tokens=100, overlap_tokens=20)
    assert len(result) > 1
    for piece in result:
        assert _count_tokens(piece) <= 100


def test_recursive_split_all_pieces_nonempty():
    long_text = "Token. " * 80
    result = _split_text_recursively(long_text, max_tokens=50, overlap_tokens=10)
    for piece in result:
        assert piece.strip(), "No empty pieces allowed"
