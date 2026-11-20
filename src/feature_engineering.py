from pathlib import Path
import pandas as pd

TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")

X_TRAIN_PATH = Path("data/processed/X_train.csv")
X_TEST_PATH = Path("data/processed/X_test.csv")
Y_TRAIN_PATH = Path("data/processed/y_train.csv")
Y_TEST_PATH = Path("data/processed/y_test.csv")

NUMERIC_COLUMNS = [
    "Age",
    "Tenure",
    "Usage Frequency",
    "Support Calls",
    "Payment Delay",
    "Total Spend",
    "Last Interaction",
]

CATEGORICAL_COLUMNS = [
    "Gender",
    "Subscription Type",
    "Contract Length",
]

FEATURE_COLUMNS = NUMERIC_COLUMNS + CATEGORICAL_COLUMNS
TARGET = "Churn"


def main() -> None:
    if not TRAIN_PATH.exists() or not TEST_PATH.exists():
        raise FileNotFoundError("Processed train.csv or test.csv not found.")

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    required = FEATURE_COLUMNS + [TARGET]
    for name, df in [("train", train_df), ("test", test_df)]:
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {name} dataset: {missing}")

    # Separate X and y; scaling and one-hot encoding are deferred into the
    # sklearn Pipeline inside model_building.py to prevent CV/test leakage.
    X_train = train_df[FEATURE_COLUMNS]
    y_train = train_df[[TARGET]]
    X_test = test_df[FEATURE_COLUMNS]
    y_test = test_df[[TARGET]]

    X_TRAIN_PATH.parent.mkdir(parents=True, exist_ok=True)
    X_train.to_csv(X_TRAIN_PATH, index=False)
    X_test.to_csv(X_TEST_PATH, index=False)
    y_train.to_csv(Y_TRAIN_PATH, index=False)
    y_test.to_csv(Y_TEST_PATH, index=False)

    print(
        f"[Feature Engineering] Saved X_train {X_train.shape}, X_test {X_test.shape} "
        f"({len(NUMERIC_COLUMNS)} numeric + {len(CATEGORICAL_COLUMNS)} categorical features)."
    )


if __name__ == "__main__":
    main()
