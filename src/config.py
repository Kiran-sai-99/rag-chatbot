"""Application settings loaded from environment variables and .env."""

from dataclasses import dataclass
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    pinecone_api_key: str
    pinecone_index_name: str
    chat_model: str
    embedding_model: str
    pinecone_cloud: str
    pinecone_region: str
    top_k: int
    min_relevance_score: float
    pdf_path: Path


def get_settings() -> Settings:
    """Read settings at call time so local environment changes take effect."""
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
        pinecone_api_key=os.getenv("PINECONE_API_KEY", "").strip(),
        pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", "agentic-ai-index").strip(),
        chat_model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini").strip(),
        embedding_model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small").strip(),
        pinecone_cloud=os.getenv("PINECONE_CLOUD", "aws").strip(),
        pinecone_region=os.getenv("PINECONE_REGION", "us-east-1").strip(),
        top_k=max(1, int(os.getenv("TOP_K", "3"))),
        min_relevance_score=min(1.0, max(0.0, float(os.getenv("MIN_RELEVANCE_SCORE", "0.35")))),
        pdf_path=Path(os.getenv("PDF_PATH", str(PROJECT_ROOT / "data" / "Ebook-Agentic-AI.pdf"))),
    )


def validate_service_keys(settings: Settings | None = None) -> Settings:
    """Return settings or raise an actionable error for missing service keys."""
    settings = settings or get_settings()
    missing = []
    if not settings.openai_api_key:
        missing.append("OPENAI_API_KEY")
    if not settings.pinecone_api_key:
        missing.append("PINECONE_API_KEY")
    if missing:
        raise RuntimeError(
            "Missing required environment variable(s): " + ", ".join(missing)
            + ". Copy .env.example to .env and add your API keys."
        )
    return settings
