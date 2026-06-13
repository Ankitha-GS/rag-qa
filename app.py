"""
Streamlit UI — upload docs, ask questions with streaming, run evals, compare chunk sizes.
Run: streamlit run app.py
"""

import json
import time
import shutil
import tempfile
from pathlib import Path

import streamlit as st

from src.rag_engine import RAGEngine, RAGConfig
from src.evaluator import RAGEvaluator, EvalCase, chunk_size_sweep

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="RAG Document Q&A", page_icon="📚", layout="wide")

st.markdown("""
<style>
.metric-card {
    background: #f8f9fa; border-radius: 8px;
    padding: 1rem; text-align: center;
}
.source-card {
    background: #f0f4ff; border-left: 3px solid #4f46e5;
    padding: 0.75rem; border-radius: 0 6px 6px 0;
    margin-bottom: 0.5rem; font-size: 0.85rem;
}
.pass-badge { color: #16a34a; font-weight: 600; }
.fail-badge { color: #dc2626; font-weight: 600; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
if "engine" not in st.session_state:
    st.session_state.engine = RAGEngine()
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📚 RAG Q&A")
    st.divider()

    st.subheader("Upload documents")
    uploaded = st.file_uploader(
        "PDF, TXT, DOCX, or MD",
        type=["pdf", "txt", "docx", "md"],
        accept_multiple_files=True,
    )

    st.subheader("Chunking settings")
    chunk_size = st.select_slider(
        "Chunk size",
        options=[200, 300, 500, 750, 1000],
        value=500,
        help="Smaller = more precise retrieval. Larger = more context per chunk.",
    )
    chunk_overlap = st.slider("Chunk overlap", 0, 200, 50, step=10)
    top_k = st.slider("Chunks to retrieve (top-K)", 1, 10, 4)
    reset_index = st.checkbox("Reset existing index on upload", value=False)

    if st.button("Index documents", type="primary", disabled=not uploaded):
        tmp_dir = Path(tempfile.mkdtemp())
        saved = []
        for f in uploaded:
            dest = tmp_dir / f.name
            dest.write_bytes(f.read())
            saved.append(str(dest))

        config = RAGConfig(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            top_k=top_k,
        )
        st.session_state.engine = RAGEngine(config=config)

        with st.spinner("Indexing…"):
            result = st.session_state.engine.index_documents(saved, reset=reset_index)

        st.session_state.indexed_files = [Path(p).name for p in saved]
        st.success(f"Indexed {result['chunks']} chunks from {result['files']} file(s)")
        if result["skipped"]:
            st.warning(f"Skipped: {result['skipped']}")

    stats = st.session_state.engine.get_collection_stats()
    st.divider()
    st.caption(f"**{stats['total_chunks']}** chunks indexed")
    if stats["sources"]:
        for s in stats["sources"]:
            st.caption(f"• {s}")


# ── Main tabs ─────────────────────────────────────────────────────────────────
tab_chat, tab_eval, tab_sweep = st.tabs(["💬 Ask", "🧪 Evaluate", "📊 Chunk sweep"])


# ── Tab 1: Chat ───────────────────────────────────────────────────────────────
with tab_chat:
    st.header("Ask your documents")

    # Render history
    for turn in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            if turn.get("sources"):
                with st.expander(f"Sources ({len(turn['sources'])} chunks)"):
                    for src in turn["sources"]:
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<b>{src['source']}</b> · page {src['page']}<br>"
                            f"<small>{src['snippet']}</small></div>",
                            unsafe_allow_html=True,
                        )
            cols = st.columns(3)
            cols[0].caption(f"⏱ {turn['latency_ms']}ms")
            cols[1].caption(f"📄 {turn['chunks_retrieved']} chunks used")

    # Input
    question = st.chat_input("Ask a question about your documents…")
    if question:
        if stats["total_chunks"] == 0:
            st.warning("Upload and index documents first.")
        else:
            with st.chat_message("user"):
                st.write(question)

            with st.chat_message("assistant"):
                placeholder = st.empty()
                full_answer = ""
                t0 = time.time()

                # Streaming output
                for token in st.session_state.engine.query_stream(question):
                    full_answer += token
                    placeholder.markdown(full_answer + "▌")

                latency = int((time.time() - t0) * 1000)
                placeholder.markdown(full_answer)

                # Fetch sources (non-streaming)
                result = st.session_state.engine.query(question)

                with st.expander(f"Sources ({len(result.sources)} chunks)"):
                    for src in result.sources:
                        st.markdown(
                            f"<div class='source-card'>"
                            f"<b>{src['source']}</b> · page {src['page']}<br>"
                            f"<small>{src['snippet']}</small></div>",
                            unsafe_allow_html=True,
                        )

                cols = st.columns(3)
                cols[0].caption(f"⏱ {latency}ms")
                cols[1].caption(f"📄 {result.chunks_retrieved} chunks used")
                cols[2].caption(f"🔢 chunk_size={st.session_state.engine.config.chunk_size}")

            st.session_state.chat_history.append({
                "question": question,
                "answer": full_answer,
                "sources": result.sources,
                "latency_ms": latency,
                "chunks_retrieved": result.chunks_retrieved,
            })

    if st.session_state.chat_history:
        if st.button("Clear chat"):
            st.session_state.chat_history = []
            st.rerun()


# ── Tab 2: Evaluate ───────────────────────────────────────────────────────────
with tab_eval:
    st.header("Evaluation — test set")
    st.caption("Measure faithfulness (no hallucination) and relevance (answers the question) against known Q&A pairs.")

    default_test_set = [
        {"question": "What is the main topic of the document?", "expected_answer": "", "category": "general"},
        {"question": "What are the key conclusions?", "expected_answer": "", "category": "general"},
        {"question": "Who are the main stakeholders mentioned?", "expected_answer": "", "category": "general"},
    ]

    test_json = st.text_area(
        "Test set (JSON array of {question, expected_answer, category})",
        value=json.dumps(default_test_set, indent=2),
        height=280,
    )

    uploaded_test = st.file_uploader("Or upload a test_set.json", type=["json"], key="eval_upload")
    if uploaded_test:
        test_json = uploaded_test.read().decode()

    if st.button("Run evaluation", type="primary"):
        if stats["total_chunks"] == 0:
            st.warning("Index documents first.")
        else:
            try:
                cases = [EvalCase(**c) for c in json.loads(test_json)]
            except Exception as e:
                st.error(f"Invalid JSON: {e}")
                st.stop()

            evaluator = RAGEvaluator(st.session_state.engine)
            progress = st.progress(0, text="Running eval cases…")
            all_results = []

            for i, case in enumerate(cases):
                r = evaluator.evaluate_case(case)
                all_results.append(r)
                progress.progress((i + 1) / len(cases), text=f"Case {i+1}/{len(cases)}")

            progress.empty()

            # Summary metrics
            avg_f = sum(r.faithfulness_score for r in all_results) / len(all_results)
            avg_r = sum(r.relevance_score for r in all_results) / len(all_results)
            passed = sum(1 for r in all_results if r.faithfulness_score >= 0.7 and r.relevance_score >= 0.7)

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Passed", f"{passed}/{len(all_results)}")
            c2.metric("Avg faithfulness", f"{avg_f:.2f}")
            c3.metric("Avg relevance", f"{avg_r:.2f}")
            c4.metric("Pass rate", f"{passed/len(all_results)*100:.0f}%")

            st.divider()

            for r in all_results:
                passed_case = r.faithfulness_score >= 0.7 and r.relevance_score >= 0.7
                badge = "✅ PASS" if passed_case else "❌ FAIL"
                with st.expander(f"{badge}  {r.question[:80]}"):
                    st.write("**Answer:**", r.actual)
                    if r.expected:
                        st.write("**Expected:**", r.expected)
                    col1, col2 = st.columns(2)
                    col1.metric("Faithfulness", f"{r.faithfulness_score:.2f}", help="Is the answer grounded in retrieved context?")
                    col2.metric("Relevance", f"{r.relevance_score:.2f}", help="Does the answer address the question?")

            # Download results
            results_json = json.dumps([r.__dict__ for r in all_results], indent=2)
            st.download_button("Download results JSON", results_json, "eval_results.json", "application/json")


# ── Tab 3: Chunk sweep ────────────────────────────────────────────────────────
with tab_sweep:
    st.header("Chunk size experiment")
    st.caption("Run the same eval against chunk sizes 200, 500, and 1000. Shows how chunking affects retrieval quality — the thing that impresses interviewers.")

    sweep_test_json = st.text_area(
        "Test set for sweep",
        value=json.dumps([
            {"question": "What is the main topic?", "expected_answer": "", "category": "general"},
            {"question": "What are the key findings?", "expected_answer": "", "category": "general"},
        ], indent=2),
        height=200,
        key="sweep_test",
    )

    sweep_sizes = st.multiselect(
        "Chunk sizes to test",
        [100, 200, 300, 500, 750, 1000],
        default=[200, 500, 1000],
    )

    if st.button("Run sweep", type="primary"):
        if stats["total_chunks"] == 0:
            st.warning("Index documents first.")
        elif len(sweep_sizes) < 2:
            st.warning("Select at least 2 chunk sizes.")
        else:
            cases = [EvalCase(**c) for c in json.loads(sweep_test_json)]
            # Use files already in session — re-index at each size
            all_source_files = [
                str(p) for p in Path("./uploads").glob("*")
                if p.suffix in (".pdf", ".txt", ".docx", ".md")
            ]
            if not all_source_files:
                st.error("No uploaded files found. Re-upload your documents first.")
            else:
                sweep_bar = st.progress(0, text="Starting sweep…")
                all_sweep = []
                for i, size in enumerate(sweep_sizes):
                    sweep_bar.progress((i) / len(sweep_sizes), text=f"Testing chunk_size={size}…")
                    config = RAGConfig(chunk_size=size, chunk_overlap=size // 10)
                    eng = RAGEngine(config=config)
                    eng.index_documents(all_source_files, reset=True)
                    ev = RAGEvaluator(eng)
                    summary = ev.run_test_set(cases)
                    summary["chunk_size"] = size
                    all_sweep.append(summary)
                sweep_bar.progress(1.0, text="Done!")

                # Results table
                import pandas as pd
                df = pd.DataFrame([{
                    "Chunk size": s["chunk_size"],
                    "Avg faithfulness": round(s["avg_faithfulness"], 3),
                    "Avg relevance": round(s["avg_relevance"], 3),
                    "Pass rate": f"{s['passed']}/{s['total']}",
                    "Avg latency (ms)": s["avg_latency_ms"],
                } for s in all_sweep])
                st.dataframe(df, use_container_width=True)

                # Download
                st.download_button(
                    "Download sweep results",
                    json.dumps(all_sweep, indent=2),
                    "sweep_results.json",
                    "application/json"
                )

                # Restore engine
                st.session_state.engine = RAGEngine()
