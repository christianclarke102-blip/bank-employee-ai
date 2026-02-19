import pickle
from pathlib import Path
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

CLEAN_PATH = Path("data/bank_employees_cleaned.parquet")
INDEX_DIR = Path("index")
INDEX_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

def row_to_doc(row: dict) -> str:
    return (
        f"Employee {row.get('First Name','')} {row.get('Last Name','')} "
        f"(Company ID {row.get('Six Digit Company ID#','')}) works in {row.get('Department','')} "
        f"as {row.get('Job Title','')} on team {row.get('Team','')}. "
        f"Branch: {row.get('Branch Location','')}. "
        f"Hire Date: {row.get('Hire Date','')}. "
        f"Monthly Salary: {row.get('Monthly Salary','')}. "
        f"Annual Salary: {row.get('Annual Salary','')}. "
        f"Age: {row.get('Age','')}. TenureYears: {row.get('TenureYears','')}."
    )

def main():
    if not CLEAN_PATH.exists():
        raise FileNotFoundError(f"Missing cleaned file: {CLEAN_PATH.resolve()}")

    df = pd.read_parquet(CLEAN_PATH)
    records = df.to_dict(orient="records")
    docs = [row_to_doc(r) for r in records]

    model = SentenceTransformer(MODEL_NAME)
    embeddings = model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)  # cosine similarity (normalized vectors)
    index.add(embeddings)

    faiss.write_index(index, str(INDEX_DIR / "employees.faiss"))
    with open(INDEX_DIR / "employees_docs.pkl", "wb") as f:
        pickle.dump({"docs": docs, "records": records}, f)

    print("✅ Index built and saved to:")
    print(" - index/employees.faiss")
    print(" - index/employees_docs.pkl")

if __name__ == "__main__":
    main()
