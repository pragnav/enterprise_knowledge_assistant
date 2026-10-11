"""
app.py  –  Enterprise Knowledge Assistant
Builds the multi-agent LangGraph RAG pipeline and exposes the CLI.

  retrieve ──► grade_docs ──► generate ──► [hallucination check]
                  │               ▲              │
                  └──► rewrite ───┘         end (if grounded)
"""
from langgraph.graph import StateGraph, END

from agents import (
    State,
    retrieve, grade_docs, generate, rewrite,
    route_after_grading, route_after_generate,
)

# ══════════════════════════════════════════════════════════════
#  BUILD LANGGRAPH WORKFLOW
# ══════════════════════════════════════════════════════════════
g = StateGraph(State)
g.add_node("retrieve",   retrieve)
g.add_node("grade_docs", grade_docs)
g.add_node("generate",   generate)
g.add_node("rewrite",    rewrite)

g.set_entry_point("retrieve")
g.add_edge("retrieve",   "grade_docs")
g.add_conditional_edges("grade_docs", route_after_grading, {
    "generate": "generate",
    "rewrite":  "rewrite",
})
g.add_edge("rewrite", "retrieve")
g.add_conditional_edges("generate", route_after_generate, {
    "end":      END,
    "generate": "generate",   # retry if hallucination detected
})

app = g.compile()

# ══════════════════════════════════════════════════════════════
#  PUBLIC API + CLI
# ══════════════════════════════════════════════════════════════

def ask(question: str) -> str:
    """Run the multi-agent RAG pipeline and return the cited answer."""
    result = app.invoke({
        "question":  question,
        "documents": [],
        "answer":    "",
        "retries":   0,
    })
    return result["answer"]


if __name__ == "__main__":
    print("Enterprise Knowledge Assistant")
    print("Agents: Retriever → Grader → Generator → Hallucination Evaluator")
    print("=" * 65)
    print("Tip: run 'python ingest.py' first to load the knowledge base.\n")
    while True:
        q = input("You: ").strip()
        if not q or q.lower() in ("quit", "exit", "q"):
            break
        print()
        print("Assistant:\n" + ask(q))
        print("-" * 65)
