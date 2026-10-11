# Enterprise Knowledge Assistant

A fully local, multi-agent RAG (Retrieval-Augmented Generation) pipeline built with LangGraph, LangChain, ChromaDB, and Ollama. Employees ask questions in plain English and receive cited answers drawn exclusively from the company knowledge base — no data leaves the network.

---

## How it works

```
You ──► retrieve ──► grade_docs ──► generate ──► hallucination check ──► Answer
                         │               ▲                │
                         └──► rewrite ───┘           (re-generate if
                                                       not grounded)
```

1. **Retrieve** – semantic search over ChromaDB returns the top-k candidate documents.
2. **Grade** – each candidate is scored for relevance to the question; irrelevant docs are dropped.
3. **Rewrite** – if no relevant docs are found, the query is rewritten and retrieval is retried (up to `MAX_RETRIES`).
4. **Generate** – a cited answer is composed from the graded documents. Returns "I don't know" if the KB has no relevant content.
5. **Hallucination check** – verifies the answer is grounded in the retrieved context; triggers regeneration if not.

---

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Add your own documents to a docs/ folder
mkdir docs
# copy .txt, .md, or .pdf files into docs/

# 3. Populate the knowledge base
python ingest.py           # uses ./docs/ if present, otherwise loads sample data

# 4. Start the assistant
python app.py
```

---

## File overview

### `config.py`
Shared configuration constants imported by all other modules. Edit this file to point at a different Ollama server, swap models, or change the ChromaDB path.

| Constant | Default | Purpose |
|---|---|---|
| `OLLAMA_URL` | `http://98.90.16.11:11434` | Remote Ollama server |
| `LLM_MODEL` | `llama3.2` | Chat model for generation & grading |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model for semantic search |
| `CHROMA_DIR` | `./chroma_db` | Local ChromaDB persistence directory |
| `MAX_RETRIES` | `2` | Max rewrite+retrieve loops before giving up |

---

### `agents.py`
Contains everything that runs *inside* the LangGraph graph:

- **LangChain components** — `llm`, `chain`, `embed`, `store`, `retriever` are initialised once here and reused by all agent nodes.
- **`State`** — the `TypedDict` that flows through the graph (`question`, `documents`, `answer`, `retries`).
- **Agent nodes** — `retrieve`, `grade_docs`, `generate`, `rewrite` — one function per agent, each receiving and returning a partial `State`.
- **Routing functions** — `route_after_grading` (generate / rewrite / no-answer) and `route_after_generate` (end / re-generate) — pure functions that inspect state and return the name of the next node.

---

### `app.py`
Wires the agents into a LangGraph `StateGraph` and exposes the public interface.

- **Graph construction** — adds nodes, sets entry point, registers conditional edges.
- **`ask(question)`** — the single public function; invokes the compiled graph and returns the cited answer string.
- **CLI loop** — `__main__` block runs an interactive terminal session.

Import `ask` from this module to embed the assistant in a web API, Slack bot, or any other integration.

---

### `ingest.py`
Populates ChromaDB with documents. Supports two modes:

- **Directory mode** — pass a folder path (default: `./docs/`) containing `.txt`, `.md`, or `.pdf` files. Documents are automatically chunked with `RecursiveCharacterTextSplitter` (500 chars, 50 overlap) before being embedded and stored.
- **Sample data fallback** — if no directory is found, loads 9 built-in documents covering HR (PTO, onboarding, benefits), IT (security, VPN, hardware requests), and Finance (expenses, budget, procurement).

Run with an optional path argument:
```bash
python ingest.py                    # uses ./docs/ or sample data
python ingest.py /path/to/kb/files  # custom directory
```

---

### `requirements.txt`
| Package | Purpose |
|---|---|
| `langchain` | Core chains and prompt utilities |
| `langchain-ollama` | Ollama LLM and embedding integrations |
| `langchain-chroma` | ChromaDB vector store integration |
| `langchain-community` | File loaders (`TextLoader`, `PyPDFLoader`) |
| `langchain-text-splitters` | `RecursiveCharacterTextSplitter` |
| `langgraph` | Multi-agent stateful graph execution |
| `chromadb` | Local vector database |
| `pypdf` | PDF parsing support |
