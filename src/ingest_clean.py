import pandas as pd
from pathlib import Path

DATA_PATH = Path("data/Bank_Employees_55_Updated.xlsx")
OUT_DIR = Path("data")
OUT_DIR.mkdir(parents=True, exist_ok=True)

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Could not find dataset at: {DATA_PATH.resolve()}")

    df = pd.read_excel(DATA_PATH)

    # Standardize column names
    df.columns = [c.strip() for c in df.columns]

    # Clean text columns
    text_cols = ["First Name", "Last Name", "Department", "Job Title", "Team", "Branch Location"]
    for c in text_cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()

    # Parse dates safely
    for c in ["Date of Birth", "Hire Date"]:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # Ensure numeric salary
    if "Monthly Salary" in df.columns:
        df["Monthly Salary"] = pd.to_numeric(df["Monthly Salary"], errors="coerce")

    # Keep Company ID as string
    if "Six Digit Company ID#" in df.columns:
        df["Six Digit Company ID#"] = df["Six Digit Company ID#"].astype("Int64").astype(str)

    # Add derived columns
    today = pd.Timestamp.today().normalize()
    if "Date of Birth" in df.columns:
        df["Age"] = ((today - df["Date of Birth"]).dt.days / 365.25).round(1)
    if "Hire Date" in df.columns:
        df["TenureYears"] = ((today - df["Hire Date"]).dt.days / 365.25).round(2)

    if "Monthly Salary" in df.columns:
        df["Annual Salary"] = (df["Monthly Salary"] * 12).round(2)

    # Save cleaned outputs
    df.to_csv(OUT_DIR / "bank_employees_cleaned.csv", index=False)
    df.to_parquet(OUT_DIR / "bank_employees_cleaned.parquet", index=False)

    print("✅ Cleaned data saved to:")
    print(" - data/bank_employees_cleaned.csv")
    print(" - data/bank_employees_cleaned.parquet")

if __name__ == "__main__":
    main()
