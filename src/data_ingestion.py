from pathlib import Path
import pandas as pd

RAW_DATA_PATH = Path("data/raw/customer_churn.csv")
INGESTED_DATA_PATH = Path("data/ingested/churn.csv")

REQUIRED_COLUMNS = [
    "CustomerID",
    "Age",
    "Gender",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Subscription Type",
    "Contract Length",
    "Total Spend",
    "Last Interaction",
    "Churn",
]

FEATURE_AND_TARGET_COLUMNS = [col for col in REQUIRED_COLUMNS if col != "CustomerID"]


def main() -> None:
    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Raw dataset not found at {RAW_DATA_PATH}")

    df = pd.read_csv(RAW_DATA_PATH)
    if df.empty:
        raise ValueError("Raw dataset is empty.")

    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in raw dataset: {missing_cols}")

    # Drop identifier column and remove any corrupted/blank records
    df = df[FEATURE_AND_TARGET_COLUMNS].dropna().reset_index(drop=True)

    # Ensure clean types for numeric features and binary target
    df["Churn"] = df["Churn"].astype(int)

    INGESTED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(INGESTED_DATA_PATH, index=False)
    print(f"[Data Ingestion] Saved {len(df):,} clean records to {INGESTED_DATA_PATH}")


if __name__ == "__main__":
    main()
