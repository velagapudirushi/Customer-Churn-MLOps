from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml

PARAMS_PATH = Path("params.yaml")
INGESTED_DATA_PATH = Path("data/ingested/churn.csv")
TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
TARGET = "Churn"


def load_params() -> dict:
    if not PARAMS_PATH.exists():
        raise FileNotFoundError(f"Parameter file not found at {PARAMS_PATH}")
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    params = load_params()
    test_size = float(params["test_size"])
    random_state = int(params["random_state"])

    if not INGESTED_DATA_PATH.exists():
        raise FileNotFoundError(f"Ingested dataset not found at {INGESTED_DATA_PATH}")

    data = pd.read_csv(INGESTED_DATA_PATH)
    if TARGET not in data.columns:
        raise ValueError(f"Target column '{TARGET}' missing from ingested data.")

    # Deduplicate exact rows before splitting to prevent train/test leakage
    initial_rows = len(data)
    data = data.drop_duplicates().reset_index(drop=True)
    deduped_rows = len(data)

    train_df, test_df = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data[TARGET],
    )

    TRAIN_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_DATA_PATH, index=False)
    test_df.to_csv(TEST_DATA_PATH, index=False)

    print(
        f"[Data Preprocessing] Deduplicated {initial_rows:,} -> {deduped_rows:,} rows | "
        f"Train: {len(train_df):,} | Test: {len(test_df):,}"
    )


if __name__ == "__main__":
    main()
