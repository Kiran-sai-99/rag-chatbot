# Agentic AI eBook RAG Chatbot

A Python Retrieval-Augmented Generation (RAG) application for asking questions about the supplied *Agentic AI* eBook. It uses PyPDF for document parsing, LangChain for chunking and model integrations, Pinecone for vector storage, LangGraph for the retrieval and answer workflow, OpenAI for embeddings and chat completions, and FastAPI for the HTTP API.

The chatbot returns an answer, the retrieved eBook excerpts with page metadata, and the best retrieved cosine similarity as a confidence signal. That score describes retrieval relevance; it is not a calibrated probability that the answer is correct. If no excerpt meets the configured relevance threshold, the system returns `I cannot answer based on the provided document.` without asking the LLM to answer.

## Architecture

1. `src/ingestion.py` loads the PDF, splits it into 1,000-character chunks with 200-character overlap, embeds each chunk with `text-embedding-3-small`, and writes vectors to the `agentic-ai-ebook` namespace in Pinecone.
2. `src/graph.py` defines a LangGraph state with the question, retrieved context, answer, and score. Its retrieval node searches the Pinecone index; its generation node refuses low-relevance questions and otherwise asks `gpt-4o-mini` to answer from the excerpts only.
3. `app.py` exposes `/chat` and `/health` through FastAPI. `/chat` returns `answer`, `retrieved_chunks`, and `confidence_score`.

## Requirements

- Python 3.10 or newer
- An OpenAI API key
- A Pinecone API key and a Pinecone serverless region supported by your account
- Network access to OpenAI and Pinecone when ingesting or asking questions

## Setup

From the project root, create and activate a virtual environment.

**Windows PowerShell:**

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

**macOS / Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and set `OPENAI_API_KEY` and `PINECONE_API_KEY`. The default Pinecone configuration uses AWS `us-east-1`; change `PINECONE_CLOUD` and `PINECONE_REGION` to a serverless region available in your Pinecone account. Keep `.env` private and do not commit it.

## Add and index the source document

Download `Ebook-Agentic-AI.pdf` from the assignment's [Google Drive file](https://drive.google.com/file/d/15VLphKcY23_fpYxN62UEQRri_psRVfP9/view) and place it at:

```text
data/Ebook-Agentic-AI.pdf
```

Run ingestion from the project root:

```bash
python -m src.ingestion
```

The script creates the configured Pinecone index if needed. It uses 1,536 dimensions and cosine distance to match `text-embedding-3-small`. Re-running ingestion adds the document chunks to the same namespace; delete that namespace in Pinecone before re-ingesting if you need to replace an earlier version of the document.

## Run the API

```bash
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000/docs` for the interactive API page. Check service availability at `GET /health` and ask a question with:

```bash
curl -X POST "http://127.0.0.1:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"query":"What is Agentic AI according to the eBook?"}'
```

Example response shape:

```json
{
  "answer": "...",
  "retrieved_chunks": [
    {
      "text": "...",
      "source": "data/Ebook-Agentic-AI.pdf",
      "page": 4,
      "relevance_score": 0.78
    }
  ],
  "confidence_score": 0.78
}
```

`confidence_score` is the best Pinecone cosine similarity returned for the question, bounded to 0–1. Tune the minimum accepted value with `MIN_RELEVANCE_SCORE` in `.env` if needed. Similarity is useful for checking retrieval quality, but should not be interpreted as answer correctness probability.

## Run the assignment benchmark queries

With the API running in another terminal:

```bash
python tests_sample_queries.py
```

The script sends six questions, including the required out-of-scope FIFA World Cup question. That last question should receive the document-grounding refusal when its retrieved excerpts are below the relevance threshold.

## API behavior

- `POST /chat` accepts `{"query": "..."}` and returns the answer, retrieved chunks, and relevance score.
- Empty or oversized queries are rejected by request validation.
- Missing API keys are reported as a service configuration error.
- Retrieval or provider errors are returned as an HTTP 502 response with a readable detail.
- The API does not retain conversation history; every request is grounded in the indexed eBook excerpts.

## Project layout

```text
.
├── data/
│   └── Ebook-Agentic-AI.pdf       # Download this from the assignment Drive link
├── src/
│   ├── __init__.py
│   ├── config.py                  # Environment configuration
│   ├── ingestion.py               # PDF loading, splitting, and Pinecone indexing
│   └── graph.py                   # LangGraph retrieval and grounded generation
├── app.py                         # FastAPI application
├── requirements.txt
├── .env.example
└── tests_sample_queries.py        # Six assignment benchmark questions
```
