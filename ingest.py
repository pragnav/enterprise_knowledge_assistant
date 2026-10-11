"""
ingest.py  –  Populate ChromaDB from files in ./docs/ or from built-in sample data.

Usage:
  python ingest.py              # loads from ./docs/ or falls back to sample data
  python ingest.py path/to/dir  # loads from a custom directory

Supported file types: .txt  .md  .pdf
"""
import os
import sys
from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import OLLAMA_URL, EMBED_MODEL, CHROMA_DIR

CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50


def load_from_directory(path: str) -> list:
    """Load and chunk .txt, .md, and .pdf files from a directory tree."""
    from langchain_community.document_loaders import TextLoader, PyPDFLoader

    loaders = {".txt": TextLoader, ".md": TextLoader, ".pdf": PyPDFLoader}
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    docs = []
    for filepath in Path(path).rglob("*"):
        loader_cls = loaders.get(filepath.suffix.lower())
        if loader_cls:
            chunks = splitter.split_documents(loader_cls(str(filepath)).load())
            docs.extend(chunks)
            print(f"  loaded {len(chunks):>3} chunks  ← {filepath}")
    return docs


# ── Built-in sample data (used when no docs/ directory is present) ──────────
SAMPLE_DOCS = [
    # ── HR ──────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Vacation & PTO Policy: Full-time employees accrue 15 days of PTO per year "
            "(20 days after 3 years of service). PTO requests must be submitted at least "
            "2 weeks in advance via the HR portal. Unused PTO up to 5 days may be carried "
            "over to the next calendar year. Sick leave is separate: 10 days per year, "
            "does not carry over."
        ),
        metadata={"source": "hr/pto-policy", "domain": "HR"},
    ),
    Document(
        page_content=(
            "Employee Onboarding: New hires complete a 3-day orientation covering company "
            "culture, tools setup, and compliance training. IT equipment is provisioned on "
            "Day 1. Employees must complete mandatory security awareness training within "
            "30 days of joining. A buddy is assigned for the first 90 days to aid integration."
        ),
        metadata={"source": "hr/onboarding", "domain": "HR"},
    ),
    Document(
        page_content=(
            "Health & Benefits: The company provides medical, dental, and vision insurance. "
            "Medical plans include HMO and PPO options; premiums are split 80/20 "
            "(company/employee). 401(k) plan with 4% employer match, vesting after 2 years. "
            "Employees may enroll or change coverage only during the annual open enrollment "
            "period held every November."
        ),
        metadata={"source": "hr/benefits", "domain": "HR"},
    ),
    # ── IT ──────────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Password & Security Policy: Passwords must be at least 12 characters and include "
            "uppercase, lowercase, numbers, and special symbols. Passwords expire every 90 days. "
            "Multi-factor authentication (MFA) is mandatory for all corporate accounts. "
            "Never share passwords. IT will never ask for your password via email or phone."
        ),
        metadata={"source": "it/security-policy", "domain": "IT"},
    ),
    Document(
        page_content=(
            "VPN & Remote Access: Use GlobalProtect VPN for all remote work sessions. "
            "Connect to vpn.company.com using your corporate credentials plus your MFA token. "
            "VPN is required whenever accessing internal systems from outside the office network. "
            "Report connectivity issues to IT helpdesk at helpdesk@company.com or call ext. 5500."
        ),
        metadata={"source": "it/vpn-access", "domain": "IT"},
    ),
    Document(
        page_content=(
            "Software & Hardware Requests: Submit all requests via the IT Service Portal at "
            "portal.it.company.com. Standard software requests are fulfilled within 3 business days. "
            "Hardware requests (laptops, monitors, peripherals) require manager approval and are "
            "fulfilled within 5-7 business days. The approved software catalog is listed on the "
            "IT intranet page."
        ),
        metadata={"source": "it/sw-hw-requests", "domain": "IT"},
    ),
    # ── Finance ─────────────────────────────────────────────────────────
    Document(
        page_content=(
            "Expense Reimbursement Policy: Submit expenses within 30 days of incurrence via Concur. "
            "Meals are reimbursable up to $75 per day with itemized receipts. "
            "Airfare must be economy class for flights under 6 hours. "
            "Hotel stays are capped at $200 per night in most cities. "
            "All expenses over $500 require manager pre-approval before being incurred."
        ),
        metadata={"source": "finance/expense-policy", "domain": "Finance"},
    ),
    Document(
        page_content=(
            "Budget Planning Cycle: Annual budgets are submitted in October for the following "
            "fiscal year. Department heads must submit budget requests to Finance by October 15. "
            "Mid-year budget reviews occur every April. "
            "Budget reallocations greater than 10% of the approved budget require CFO approval."
        ),
        metadata={"source": "finance/budget-cycle", "domain": "Finance"},
    ),
    Document(
        page_content=(
            "Procurement Policy: All vendor contracts above $10,000 must go through the "
            "Procurement team. Use the approved vendor list whenever possible to speed up processing. "
            "Onboarding a new vendor requires a vendor risk assessment and legal review, typically "
            "taking 2-3 weeks. Purchase orders are required for all purchases over $1,000."
        ),
        metadata={"source": "finance/procurement", "domain": "Finance"},
    ),
]  # end SAMPLE_DOCS

if __name__ == "__main__":
    docs_dir = sys.argv[1] if len(sys.argv) > 1 else "./docs"

    if os.path.isdir(docs_dir):
        print(f"Loading documents from '{docs_dir}' ...")
        docs = load_from_directory(docs_dir)
        if not docs:
            print("No supported files found – falling back to sample data.")
            docs = SAMPLE_DOCS
        else:
            print(f"Loaded {len(docs)} chunks total.")
    else:
        print(f"'{docs_dir}' not found – using built-in sample documents.")
        docs = SAMPLE_DOCS

    print(f"Connecting to Ollama at {OLLAMA_URL} ...")
    embed = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
    store = Chroma(
        collection_name="enterprise_kb",
        embedding_function=embed,
        persist_directory=CHROMA_DIR,
    )
    store.add_documents(docs)
    print(f"Ingested {len(docs)} documents into ChromaDB at '{CHROMA_DIR}'")
    print("Run:  python app.py")
