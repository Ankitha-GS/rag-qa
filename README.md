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
Indexing (once):   PDF → chunk → embed → ChromaDB
Querying (live):   Question → embed → top-K retrieval → LLM → streamed answer
```

**Tech stack:**
- Langchain for orchestration and document loading
- ChromaDB as the local vector store (swap to Pinecone for production scale)
- OpenAI `text-embedding-3-small` for embeddings (~$0.00002 per 1K tokens)
- `gpt-4o-mini` as the generator (cheap, fast, good at grounding)
- Streamlit for the UI, FastAPI for the REST/streaming API

---

## Eval results

Tested on a 6-question set covering general comprehension, factual recall, and hallucination resistance:

| Metric               | Score  |
|----------------------|--------|
| Avg faithfulness     | 0.91   |
| Avg relevance        | 0.88   |
| Pass rate (≥0.7 both)| 5/6    |
| Avg latency          | 1340ms |
| Hallucination test   | PASS   |

*Faithfulness: is every claim supported by retrieved context? Relevance: does the answer address the question? Both scored 0–1 by a judge LLM.*

---

## Chunk size experiment

Running the same 6 questions at three different chunk sizes reveals a clear trade-off:

| Chunk size | Faithfulness | Relevance | Avg latency | Chunks indexed |
|-----------|-------------|-----------|-------------|----------------|
| 200       | 0.84        | 0.79      | 980ms       | 847            |
| 500       | 0.91        | 0.88      | 1340ms      | 342            |
| 1000      | 0.86        | 0.82      | 1480ms      | 178            |

**Finding:** chunk_size=500 with overlap=50 gave the best faithfulness/relevance balance on this corpus. Smaller chunks retrieved more precise passages but sometimes missed context that spanned paragraph boundaries. Larger chunks had lower relevance because the LLM received noisier context.

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
└── data/               # Drop your documents here
```

---

## Quickstart

```bash
git clone https://github.com/you/rag-qa
cd rag-qa
pip install -r requirements.txt

# Add your key
echo "OPENAI_API_KEY=sk-..." > .env

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

**Why ChromaDB instead of Pinecone?** Zero infra cost for a portfolio project. The abstraction is identical — swap one line to use Pinecone for production.

**Why `text-embedding-3-small`?** 3x cheaper than ada-002, better quality, 1536 dimensions. At portfolio scale the cost is negligible.

**Why chunk overlap?** Prevents information loss at chunk boundaries. A sentence that straddles two chunks would otherwise be split in half and retrieved poorly.

**Why a custom prompt?** The default LangChain prompt doesn't explicitly forbid hallucination. The custom prompt forces the model to say "I don't know" rather than guess — testable with the hallucination eval case.

**What I'd change at production scale:**
- Reranker (Cohere rerank) after initial retrieval to improve top-K precision
- Hybrid search (BM25 + vector) for better keyword recall
- Async indexing pipeline with a job queue
- User-level document namespacing

---

## What the eval framework measures

`faithfulness` — does the answer contain only claims supported by retrieved chunks? Scored by a judge LLM that reads the context and the answer.

`relevance` — does the answer address the question asked? Scored independently of faithfulness.

A perfect RAG system scores 1.0 on both. The hallucination test case (`expected_answer: "I don't know..."`) specifically tests whether the model stays grounded when the answer isn't in the documents.
