"""Parse the source PDF, split it into chunks, and index it in Pinecone."""

from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from src.config import get_settings, validate_service_keys

INDEX_DIMENSION = 1536


def ensure_index(index_name: str, api_key: str, cloud: str, region: str) -> None:
    """Create the cosine index if it does not already exist."""
    client = Pinecone(api_key=api_key)
    if index_name not in client.list_indexes().names():
        client.create_index(
            name=index_name,
            dimension=INDEX_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=cloud, region=region),
        )


def run_ingestion(pdf_path: str | Path | None = None, index_name: str | None = None) -> int:
    """Index all non-empty PDF chunks and return the number of chunks written."""
    settings = validate_service_keys()
    source = Path(pdf_path) if pdf_path else settings.pdf_path
    if not source.is_file():
        raise FileNotFoundError(
            f"Source PDF not found at {source}. Download Ebook-Agentic-AI.pdf "
            "from the assignment's Google Drive link and save it in data/."
        )

    docs = PyPDFLoader(str(source)).load()
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    ).split_documents(docs)
    chunks = [chunk for chunk in chunks if chunk.page_content.strip()]
    if not chunks:
        raise ValueError(f"No readable text was found in {source}.")

    name = index_name or settings.pinecone_index_name
    ensure_index(name, settings.pinecone_api_key, settings.pinecone_cloud, settings.pinecone_region)
    embeddings = OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
    )
    PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=name,
        pinecone_api_key=settings.pinecone_api_key,
        namespace="agentic-ai-ebook",
    )
    return len(chunks)


if __name__ == "__main__":
    count = run_ingestion()
    print(f"Indexed {count} chunks from the Agentic AI eBook.")
