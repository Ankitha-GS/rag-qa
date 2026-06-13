"""
Evaluation framework — runs a test set against the RAG pipeline and
measures faithfulness, answer relevance, and chunk-size sensitivity.

Usage:
    python -m src.evaluator --test-file evals/test_set.json
    python -m src.evaluator --sweep-chunks   # compare chunk sizes 200/500/1000
"""
import os
import json
import time
import argparse
import statistics
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List

from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate

from .rag_engine import RAGEngine, RAGConfig


@dataclass
class EvalCase:
    question: str
    expected_answer: str
    category: str = "general"


@dataclass
class EvalResult:
    question: str
    expected: str
    actual: str
    faithfulness_score: float    # 0-1: is answer grounded in retrieved chunks?
    relevance_score: float       # 0-1: does answer address the question?
    exact_match: bool
    latency_ms: int
    chunks_retrieved: int

FAITHFULNESS_PROMPT = PromptTemplate(
    input_variables=["answer", "context"],
    template="""You are an evaluation judge. Score whether this answer is supported by the context.
Return ONLY a single decimal number between 0.0 and 1.0. No explanation, no text, just the number.

Scoring:
1.0 = answer is fully supported by the context
0.7 = answer is mostly supported with minor gaps
0.5 = answer is partially supported
0.2 = answer has claims not in context
0.0 = answer is completely unsupported or hallucinates

Context: {context}
Answer: {answer}
Score (just the number):"""
)

RELEVANCE_PROMPT = PromptTemplate(
    input_variables=["question", "answer"],
    template="""You are an evaluation judge. Score how well this answer addresses the question.
Return ONLY a single decimal number between 0.0 and 1.0. No explanation, no text, just the number.

Scoring:
1.0 = answer directly and completely addresses the question
0.7 = answer addresses the question but with some extra or missing info
0.5 = answer partially addresses the question
0.2 = answer barely addresses the question
0.0 = answer does not address the question at all

Examples of good answers:
- Question: "What is the CGPA?" Answer: "9.2" → Score: 1.0
- Question: "Which college?" Answer: "Sai Vidya Institute of Technology" → Score: 1.0
- Question: "What is the capital of France?" Answer: "I don't know based on the provided documents" → Score: 1.0 (correct refusal)

Question: {question}
Answer: {answer}
Score (just the number):"""
)





class RAGEvaluator:
    def __init__(self, engine: RAGEngine):
        self.engine = engine
        self.judge_llm = ChatGroq(
            model="llama-3.1-8b-instant",
            api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

    def _score(self, prompt_template: PromptTemplate, **kwargs) -> float:
        """Call the judge LLM and parse a 0-1 float score."""
        prompt = prompt_template.format(**kwargs)
        response = self.judge_llm.invoke(prompt)
        try:
            return min(1.0, max(0.0, float(response.content.strip())))
        except ValueError:
            return 0.0

    def evaluate_case(self, case: EvalCase) -> EvalResult:
        t0 = time.time()
        result = self.engine.query(case.question)
        latency = int((time.time() - t0) * 1000)

        # Retrieve context string for faithfulness scoring
        if self.engine.vectorstore:
            docs = self.engine.vectorstore.as_retriever(
                search_kwargs={"k": self.engine.config.top_k}
            ).invoke(case.question)
            context = "\n\n".join(d.page_content for d in docs)
        else:
            context = ""

        faithfulness = self._score(
            FAITHFULNESS_PROMPT, answer=result.answer, context=context
        )
        relevance = self._score(
            RELEVANCE_PROMPT, question=case.question, answer=result.answer
        )
        exact = case.expected_answer.lower().strip() in result.answer.lower()

        return EvalResult(
            question=case.question,
            expected=case.expected_answer,
            actual=result.answer,
            faithfulness_score=faithfulness,
            relevance_score=relevance,
            exact_match=exact,
            latency_ms=latency,
            chunks_retrieved=result.chunks_retrieved,
        )

    def run_test_set(self, cases: List[EvalCase]) -> dict:
        """Run all test cases and return aggregate metrics."""
        results = []
        print(f"\nRunning {len(cases)} eval cases...")
        for i, case in enumerate(cases, 1):
            r = self.evaluate_case(case)
            time.sleep(5)  # wait 5 seconds between calls to stay under rate limit
            results.append(r)
            status = "PASS" if r.faithfulness_score >= 0.7 and r.relevance_score >= 0.7 else "FAIL"
            print(f"  [{i}/{len(cases)}] {status}  F={r.faithfulness_score:.2f}  R={r.relevance_score:.2f}  {case.question[:60]}")

        faithfulness_scores = [r.faithfulness_score for r in results]
        relevance_scores = [r.relevance_score for r in results]
        latencies = [r.latency_ms for r in results]

        summary = {
            "total": len(results),
            "passed": sum(1 for r in results if r.faithfulness_score >= 0.7 and r.relevance_score >= 0.7),
            "avg_faithfulness": round(statistics.mean(faithfulness_scores), 3),
            "avg_relevance": round(statistics.mean(relevance_scores), 3),
            "avg_latency_ms": round(statistics.mean(latencies)),
            "p95_latency_ms": round(sorted(latencies)[int(len(latencies) * 0.95)]),
            "exact_match_rate": round(sum(r.exact_match for r in results) / len(results), 3),
            "config": asdict(self.engine.config),
            "results": [asdict(r) for r in results],
        }
        return summary


def chunk_size_sweep(file_paths: List[str], test_cases: List[EvalCase], sizes=(200, 500, 1000)):
    """
    Run the same eval set against three different chunk sizes.
    This is the experiment that impresses interviewers.
    """
    sweep_results = []
    for size in sizes:
        print(f"\n{'='*50}")
        print(f"Chunk size: {size}")
        config = RAGConfig(chunk_size=size, chunk_overlap=size // 10)
        engine = RAGEngine(config=config)
        stats = engine.index_documents(file_paths, reset=True)
        print(f"  Indexed {stats['chunks']} chunks from {stats['files']} files")

        evaluator = RAGEvaluator(engine)
        summary = evaluator.run_test_set(test_cases)
        summary["chunk_size"] = size
        summary["total_chunks_indexed"] = stats["chunks"]
        sweep_results.append(summary)

    print("\n\n" + "="*50)
    print("CHUNK SIZE SWEEP RESULTS")
    print("="*50)
    print(f"{'Chunk':>8}  {'Faithful':>10}  {'Relevant':>10}  {'Pass%':>8}  {'Avg Lat':>10}  {'Chunks Idx':>12}")
    for r in sweep_results:
        pct = round(r["passed"] / r["total"] * 100)
        print(f"{r['chunk_size']:>8}  {r['avg_faithfulness']:>10.3f}  {r['avg_relevance']:>10.3f}  {pct:>7}%  {r['avg_latency_ms']:>9}ms  {r['total_chunks_indexed']:>12}")

    return sweep_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-file", default="evals/test_set.json")
    parser.add_argument("--docs-dir", default="data/")
    parser.add_argument("--sweep-chunks", action="store_true")
    parser.add_argument("--output", default="evals/results.json")
    args = parser.parse_args()

    # Load test cases
    with open(args.test_file) as f:
        raw = json.load(f)
    cases = [EvalCase(**c) for c in raw]

    # Find documents
    data_path = Path(args.docs_dir)
    file_paths = [str(p) for p in data_path.glob("*") if p.suffix in (".pdf", ".txt", ".docx", ".md")]

    if args.sweep_chunks:
        results = chunk_size_sweep(file_paths, cases)
        Path(args.output).parent.mkdir(exist_ok=True)
        with open(args.output.replace(".json", "_sweep.json"), "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output.replace('.json', '_sweep.json')}")
    else:
        engine = RAGEngine()
        if file_paths:
            engine.index_documents(file_paths)
        evaluator = RAGEvaluator(engine)
        summary = evaluator.run_test_set(cases)

        print(f"\nFinal: {summary['passed']}/{summary['total']} passed")
        print(f"Avg faithfulness: {summary['avg_faithfulness']}")
        print(f"Avg relevance:    {summary['avg_relevance']}")

        Path(args.output).parent.mkdir(exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Results saved to {args.output}")
