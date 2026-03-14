"""Phase 1 smoke tests."""
from stripe_rag.config import Settings, get_settings


def test_settings_defaults():
    settings = get_settings()
    assert settings.eval_embedding_model == "text-embedding-3-small"
    assert settings.openai_embedding_model == "text-embedding-3-large"
    assert settings.qdrant_collection_name == "stripe_docs"
    assert settings.retrieval_final_top_k == 25
    assert settings.langsmith_project == "stripe-rag-chatbot"


def test_settings_paths():
    settings = get_settings()
    assert settings.raw_html_dir.name == "raw_html"
    assert settings.markdown_dir.name == "markdown"
    assert settings.documents_jsonl.suffix == ".jsonl"
    assert settings.chunks_jsonl.suffix == ".jsonl"