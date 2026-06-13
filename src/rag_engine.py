import os
import time
from pathlib import Path
from typing import Generator, List, Optional
from dataclasses import dataclass, field

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

load_dotenv()


@dataclass
class RAGConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    top_k: int = 4
    model: str = "llama-3.1-8b-instant"
    embedding_model: str = "all-MiniLM-L6-v2"
    temperature: float = 0
    persist_dir: str = "./chroma_db"


@dataclass
class QueryResult:
    answer: str
    sources: List[dict]
    latency_ms: int
    chunks_retrieved: int
    config: dict = field(default_factory=dict)


LOADER_MAP = {
    ".pdf":  PyPDFLoader,
    ".txt":  TextLoader,
    ".docx": Docx2txtLoader,
}

RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a precise document assistant. Answer ONLY using the context provided.
If the answer is not present in the context, respond with exactly:
"I don't know based on the provided documents."

Do not speculate or draw on outside knowledge.

Context:
{context}

Question: {question}
Answer:"""
)


class RAGEngine:
    def __init__(self, config: RAGConfig = None):
        self.config = config or RAGConfig()
        self.embeddings = HuggingFaceEmbeddings(
            model_name=self.config.embedding_model,
            cache_folder="./models"
        )
        self.vectorstore: Optional[Chroma] = None
        self._load_existing_store()

    def _load_existing_store(self):
        if Path(self.config.persist_dir).exists():
            self.vectorstore = Chroma(
                persist_directory=self.config.persist_dir,
                embedding_function=self.embeddings
            )

    def index_documents(self, file_paths: List[str], reset: bool = False) -> dict:
        if reset:
            if self.vectorstore is not None:
                self.vectorstore = None
            import gc
            gc.collect()
            import time
            time.sleep(2)
            # Use unique folder per run to avoid Windows file lock
            self.config.persist_dir = f"./chroma_sweep_{self.config.chunk_size}"

        all_chunks = []
        skipped = []
        for path in file_paths:
            suffix = Path(path).suffix.lower()
            loader_cls = LOADER_MAP.get(suffix)
            if not loader_cls:
                skipped.append(path)
                continue
            try:
                docs = loader_cls(path).load()
                for doc in docs:
                    doc.metadata["source"] = Path(path).name
                chunks = splitter.split_documents(docs)
                all_chunks.extend(chunks)
            except Exception as e:
                skipped.append(f"{path} ({e})")

        if not all_chunks:
            return {"chunks": 0, "files": 0, "skipped": skipped}

        self.vectorstore = Chroma.from_documents(
            documents=all_chunks,
            embedding=self.embeddings,
            persist_directory=self.config.persist_dir
        )

        return {
            "chunks": len(all_chunks),
            "files": len(file_paths) - len(skipped),
            "skipped": skipped,
            "chunk_size": self.config.chunk_size,
            "chunk_overlap": self.config.chunk_overlap,
        }

    def _get_llm(self, streaming=False):
        return ChatGroq(
            model=self.config.model,
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=self.config.temperature,
            streaming=streaming,
        )

    def query(self, question: str) -> QueryResult:
        if not self.vectorstore:
            return QueryResult(
                answer="No documents indexed yet.",
                sources=[], latency_ms=0, chunks_retrieved=0
            )

        t0 = time.time()
        docs = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.top_k}
        ).invoke(question)
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        llm = self._get_llm()
        prompt_text = RAG_PROMPT.format(context=context, question=question)
        response = llm.invoke(prompt_text)

        latency = int((time.time() - t0) * 1000)
        sources = [
            {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", "?"),
                "snippet": d.page_content[:200] + "..."
            }
            for d in docs
        ]

        return QueryResult(
            answer=response.content,
            sources=sources,
            latency_ms=latency,
            chunks_retrieved=len(docs),
        )

    def query_stream(self, question: str) -> Generator[str, None, None]:
        if not self.vectorstore:
            yield "No documents indexed yet."
            return

        docs = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.top_k}
        ).invoke(question)
        context = "\n\n---\n\n".join(d.page_content for d in docs)

        llm = self._get_llm(streaming=True)
        prompt_text = RAG_PROMPT.format(context=context, question=question)
        for chunk in llm.stream(prompt_text):
            yield chunk.content

    def get_collection_stats(self) -> dict:
        if not self.vectorstore:
            return {"total_chunks": 0, "sources": []}
        collection = self.vectorstore._collection
        count = collection.count()
        metadatas = collection.get(include=["metadatas"])["metadatas"]
        sources = list({m.get("source", "unknown") for m in metadatas})
        return {"total_chunks": count, "sources": sources}