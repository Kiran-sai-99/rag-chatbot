"""FastAPI interface for the Agentic AI eBook RAG chatbot."""

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.graph import answer_question, build_rag_graph

app = FastAPI(
    title="Agentic AI RAG API",
    description="Answers questions using the Agentic AI eBook indexed in Pinecone.",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000, description="Question about the eBook")


class RetrievedChunk(BaseModel):
    text: str
    source: str
    page: int | None = None
    relevance_score: float


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: list[RetrievedChunk]
    confidence_score: float = Field(ge=0, le=1, description="Top cosine relevance score; a retrieval signal, not a probability")


@lru_cache(maxsize=1)
def _graph():
    return build_rag_graph()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    try:
        return answer_question(request.query.strip(), _graph())
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"RAG request failed: {exc}") from exc
