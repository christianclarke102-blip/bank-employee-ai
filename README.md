# Bank Employee AI Q&A System (Local + Free)

This project builds a fully local AI system that:

- Cleans a banking employee dataset
- Generates derived features (age, tenure, annual salary)
- Builds a semantic search index using FAISS
- Uses a local LLM (Ollama) for natural-language querying

## Tech Stack
- Python
- Pandas
- FAISS
- Sentence Transformers
- Ollama (Llama 3.2)
- Streamlit (optional UI)

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
ollama pull llama3.2:3b
