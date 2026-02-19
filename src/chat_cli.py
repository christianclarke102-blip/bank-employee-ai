import pickle
from pathlib import Path
import faiss
from sentence_transformers import SentenceTransformer
from ollama import Client

INDEX_DIR = Path("index")
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3.2:3b"

def load_assets():
    index = faiss.read_index(str(INDEX_DIR / "employees.faiss"))
    with open(INDEX_DIR / "employees_docs.pkl", "rb") as f:
        data = pickle.load(f)
    embedder = SentenceTransformer(MODEL_NAME)
    return index, data, embedder

def retrieve(query, index, data, embedder, k=10):
    q_emb = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    scores, ids = index.search(q_emb, k)
    hits = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:
            continue
        hits.append((float(score), data["docs"][idx]))
    return hits

def main():
    index, data, embedder = load_assets()
    client = Client(host="http://localhost:11434")

    print("✅ Ready. Ask questions about the dataset. Type 'exit' to quit.")
    print("Tip: Type '/show' after a question to display the retrieved rows.\n")

    last_hits = None

    while True:
        q = input("You: ").strip()
        if q.lower() in {"exit", "quit"}:
            break

        # Special command: show retrieved rows from the last question
        if q.lower() == "/show":
            if not last_hits:
                print("\n(No retrieved rows yet — ask a question first.)\n")
            else:
                print("\nTop retrieved rows (evidence):")
                for i, (s, doc) in enumerate(last_hits, 1):
                    print(f"{i}. score={s:.3f} | {doc}")
                print()
            continue

        hits = retrieve(q, index, data, embedder, k=10)
        last_hits = hits

        context = "\n".join([f"- {h[1]}" for h in hits])

        prompt = f"""
You are a dataset Q&A assistant.

RULES (follow strictly):
1) ONLY use facts from the CONTEXT below.
2) If the answer is not explicitly in the CONTEXT, respond: "Not found in the retrieved dataset rows."
3) Do NOT invent departments, names, or numbers.
4) When listing results, quote employee names exactly as shown in CONTEXT.

CONTEXT:
{context}

QUESTION:
{q}

Now answer:
"""

        resp = client.chat(
            model=OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": "Answer strictly from provided context; never guess."},
                {"role": "user", "content": prompt},
            ],
        )
        print("\nAssistant:", resp["message"]["content"].strip(), "\n")

if __name__ == "__main__":
    main()
