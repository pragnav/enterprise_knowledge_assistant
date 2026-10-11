"""
agents.py  –  LangChain components, graph State, agent nodes, and routing functions.
"""
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import END
from typing import TypedDict

from config import OLLAMA_URL, LLM_MODEL, EMBED_MODEL, CHROMA_DIR, MAX_RETRIES

# ── Shared components ──────────────────────────────────────────
llm       = ChatOllama(model=LLM_MODEL, base_url=OLLAMA_URL, temperature=0)
chain     = llm | StrOutputParser()
embed     = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
store     = Chroma(collection_name="enterprise_kb", embedding_function=embed,
                   persist_directory=CHROMA_DIR)
retriever = store.as_retriever(search_kwargs={"k": 5})

# ── Graph state ────────────────────────────────────────────────
class State(TypedDict):
    question : str
    documents: list   # list[Document]
    answer   : str
    retries  : int

# ══════════════════════════════════════════════════════════════
#  AGENT NODES
# ══════════════════════════════════════════════════════════════

def retrieve(state: State) -> dict:
    """Retrieval Agent – semantic search over ChromaDB."""
    print("  [retriever] searching knowledge base …")
    docs = retriever.invoke(state["question"])
    print(f"  [retriever] {len(docs)} candidate docs found")
    return {"documents": docs}


def grade_docs(state: State) -> dict:
    """Grader Agent – keep only documents relevant to the question."""
    print("  [grader]    scoring relevance …")
    relevant = []
    for doc in state["documents"]:
        resp = chain.invoke([HumanMessage(content=(
            "Is this document relevant to the question? Reply ONLY 'yes' or 'no'.\n"
            f"Question: {state['question']}\n"
            f"Document: {doc.page_content[:400]}"
        ))])
        if "yes" in resp.lower():
            relevant.append(doc)
    print(f"  [grader]    {len(relevant)}/{len(state['documents'])} docs passed")
    return {"documents": relevant}


def generate(state: State) -> dict:
    """Generator Agent – produce a cited answer from graded documents."""
    print("  [generator] composing answer …")
    docs = state["documents"]

    if not docs:
        return {
            "answer": "I don't have enough information in the knowledge base to answer that question.",
            "retries": MAX_RETRIES,
        }

    context = "\n\n".join(f"[{i+1}] {d.page_content}" for i, d in enumerate(docs))
    sources = [d.metadata.get("source", f"doc-{i+1}") for i, d in enumerate(docs)]

    answer = chain.invoke([HumanMessage(content=(
        "You are an enterprise knowledge assistant. "
        "Answer the employee's question using ONLY the numbered context below. "
        "Cite every fact with its source number in square brackets, e.g. [1] or [2].\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state['question']}\n\nAnswer:"
    ))])

    if sources:
        refs    = "\n".join(f"  [{i+1}] {s}" for i, s in enumerate(sources))
        answer += f"\n\nSources:\n{refs}"

    return {"answer": answer, "retries": state.get("retries", 0) + 1}


def rewrite(state: State) -> dict:
    """Query Rewriter – refine query when retrieval returns no relevant docs."""
    new_q = chain.invoke([HumanMessage(content=(
        "Rewrite the following question to improve enterprise document retrieval. "
        "Return ONLY the rewritten question, nothing else.\n"
        f"Original: {state['question']}"
    ))])
    new_q = new_q.strip()
    print(f"  [rewriter]  → {new_q}")
    return {"question": new_q, "retries": state.get("retries", 0) + 1}

# ══════════════════════════════════════════════════════════════
#  ROUTING / EVALUATOR FUNCTIONS
# ══════════════════════════════════════════════════════════════

def route_after_grading(state: State) -> str:
    """Router: decide whether to generate, rewrite, or force-generate."""
    if state["documents"]:
        return "generate"
    if state.get("retries", 0) < MAX_RETRIES:
        print("  [router]    no relevant docs – rewriting query …")
        return "rewrite"
    print("  [router]    max retries reached – generating best-effort answer")
    return "generate"


def route_after_generate(state: State) -> str:
    """Hallucination Evaluator – verify answer is grounded in retrieved context."""
    if state.get("retries", 0) >= MAX_RETRIES or not state["documents"]:
        return "end"
    context = " ".join(d.page_content[:200] for d in state["documents"])
    verdict = chain.invoke([HumanMessage(content=(
        "Is this answer fully supported by the context provided? Reply ONLY 'yes' or 'no'.\n"
        f"Context: {context}\n"
        f"Answer: {state['answer']}"
    ))])
    grounded = "yes" in verdict.lower()
    print(f"  [halluc-eval] grounded={grounded}")
    return "end" if grounded else "generate"
