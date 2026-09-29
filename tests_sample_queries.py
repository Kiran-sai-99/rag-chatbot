"""Assignment benchmark queries for POST /chat."""

import json
import os
from urllib.request import Request, urlopen

API_URL = os.getenv("RAG_API_URL", "http://127.0.0.1:8000/chat")
QUERIES = [
    "What is Agentic AI according to the eBook?",
    "How do AI agents differ from traditional automation systems?",
    "What are the core components of an Agentic Architecture?",
    "What role does memory play in Agentic AI workflows?",
    "Who won the 2022 FIFA World Cup?",
    "What challenges do multi-agent systems help businesses address?",
]


if __name__ == "__main__":
    for query in QUERIES:
        request = Request(
            API_URL,
            data=json.dumps({"query": query}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urlopen(request, timeout=120) as response:
            result = json.loads(response.read().decode("utf-8"))
        print(json.dumps({"query": query, **result}, indent=2, ensure_ascii=False))
