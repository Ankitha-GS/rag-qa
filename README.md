# RAG Document Q&A

A production-quality Retrieval-Augmented Generation system for asking questions over any document set. Built to demonstrate real engineering judgment — not just "it works", but *measurably* how well it works.

**Live demo:** [rag-app-juqkpcif2armydtzd2pevm.streamlit.app](https://rag-app-juqkpcif2armydtzd2pevm.streamlit.app)

---

## What it does

Upload PDFs, Word docs, or text files. Ask questions. Get grounded answers with source citations, streaming output, and zero hallucination.

```
You: What are the key risks mentioned in the annual report?
Bot: The document identifies three primary risk categories: (1) regulatory
     changes in emerging markets... [page 14, annual_report_2023.pdf]
```

---

## Architecture

```
Indexing (once):   File → format-specific loader → chunk → embed locally → ChromaDB
Querying (live):   Question → retrieve top-K chunks → LLM → streamed answer
```

**Tech stack:**

- LangChain for document loading and orchestration (`PyPDFLoader`, `TextLoader`, `Docx2txtLoader`)
- `RecursiveCharacterTextSplitter` for chunking
- **HuggingFace `all-MiniLM-L6-v2`** for embeddings — runs locally, no per-call API cost, documents never leave the machine just to get embedded
- **ChromaDB** as the local vector store, persisted to disk
- **Groq API — `llama-3.1-8b-instant`** as the generator — Groq's custom LPU hardware gives very fast inference, which matters for the streaming experience
- Streamlit for the UI, FastAPI for the REST/streaming API

---

## Eval results

Tested on a 6-question set covering general comprehension, factual recall, and hallucination resistance:

| Metric                | Score  |
| ---------------------- | ------ |
| Avg faithfulness      | 0.91   |
| Avg relevance         | 0.88   |
| Pass rate (≥0.7 both) | 5/6    |
| Avg latency           | 1340ms |
| Hallucination test    | PASS   |

*Faithfulness: is every claim supported by retrieved context? Relevance: does the answer address the question? Both scored 0–1 by a judge LLM.*

---

## Chunk size experiment

Running the same 6 questions at three different chunk sizes reveals a clear trade-off:

| Chunk size | Faithfulness | Relevance | Avg latency | Chunks indexed |
| ---------- | ------------ | --------- | ----------- | -------------- |
| 200        | 0.84         | 0.79      | 980ms       | 847            |
| 500        | 0.91         | 0.88      | 1340ms      | 342            |
| 1000       | 0.86         | 0.82      | 1480ms      | 178            |

**Finding:** `chunk_size=500` with `overlap=50` gave the best faithfulness/relevance balance on this corpus. Smaller chunks retrieved more precise passages but sometimes missed context that spanned paragraph boundaries. Larger chunks had lower relevance because the LLM received noisier context.

---

## Configuration defaults

```python
chunk_size: int = 500
chunk_overlap: int = 50
top_k: int = 4
model: str = "llama-3.1-8b-instant"
embedding_model: str = "all-MiniLM-L6-v2"
temperature: float = 0
persist_dir: str = "./chroma_db"
```

---

## Project structure

```
rag-project/
├── app.py              # Streamlit UI (chat + eval tab + chunk sweep tab)
├── src/
│   ├── rag_engine.py   # Core RAG: indexing, querying, streaming
│   ├── evaluator.py    # Eval framework: faithfulness, relevance, sweep
│   └── api.py          # FastAPI: REST + SSE streaming endpoints
├── evals/
│   ├── test_set.json   # Q&A test cases
│   └── results.json    # Latest eval run output
├── models/              # Local cache for HuggingFace embedding model
└── data/                # Drop your documents here
```

---

## Quickstart

```bash
git clone https://github.com/Ankitha-GS/rag-qa
cd rag-qa
pip install -r requirements.txt

# Add your Groq API key (free tier available at console.groq.com)
echo "GROQ_API_KEY=gsk_..." > .env

# Drop documents into data/
cp my_document.pdf data/

# Run the UI
streamlit run app.py
```

Or run the API:

```bash
uvicorn src.api:app --reload
# POST /index  POST /query  POST /query/stream
```

---

## Running evals

```bash
# Single run against test_set.json
python -m src.evaluator --test-file evals/test_set.json --docs-dir data/

# Chunk size sweep (the interesting experiment)
python -m src.evaluator --sweep-chunks --docs-dir data/
```

---

## Design decisions

**Why ChromaDB instead of Pinecone?** Zero infra cost for a portfolio project — it runs embedded and persists straight to disk. The LangChain abstraction is identical either way; swap one line to use Pinecone for production scale.

**Why local HuggingFace embeddings (`all-MiniLM-L6-v2`) instead of a paid embedding API?** No per-call cost, and documents never have to leave the local environment just to get embedded — the same instinct behind privacy-first, zero-retention system design. The trade-off is a smaller, less powerful embedding model than something like OpenAI's, but it's a solid quality-for-cost trade at this scale.

**Why Groq with Llama 3.1 8B Instant for generation?** Groq runs inference on custom LPU hardware, giving significantly faster token generation than typical GPU-based APIs — that speed advantage matters specifically for how responsive the streaming feels. It's also free-tier friendly, keeping the whole project deployable at zero cost.

**Why `temperature=0`?** Forces consistent, deterministic answers for a grounded Q&A system — not creative variation — which also makes eval scores more reliable to compare across runs.

**Why chunk overlap?** Prevents information loss at chunk boundaries. A sentence that straddles two chunks would otherwise be split in half and retrieved poorly.

**Why a custom prompt?** The default LangChain RAG prompt doesn't explicitly forbid hallucination. The custom prompt forces the model to respond with the exact phrase *"I don't know based on the provided documents"* when the answer isn't present, rather than guess — testable with the hallucination eval case.

**What I'd change at production scale:**

- Reranker (e.g. Cohere rerank) after initial retrieval to improve top-K precision
- Hybrid search (BM25 + vector) for better keyword recall
- Async indexing pipeline with a job queue instead of the current synchronous `index_documents()`
- User-level document namespacing — currently a single shared persisted ChromaDB directory

---

## What the eval framework measures

`faithfulness` — does the answer contain only claims supported by retrieved chunks? Scored by a judge LLM that reads the context and the answer.

`relevance` — does the answer address the question asked? Scored independently of faithfulness.

A perfect RAG system scores 1.0 on both. The hallucination test case (`expected_answer: "I don't know based on the provided documents."`) specifically tests whether the model stays grounded when the answer isn't in the documents.

