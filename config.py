"""
config.py  –  Shared configuration constants.
"""
OLLAMA_URL  = "http://98.90.16.11:11434"
LLM_MODEL   = "llama3.2"
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR  = "./chroma_db"
MAX_RETRIES = 2           # max rewrite+retrieve loops before forcing an answer
