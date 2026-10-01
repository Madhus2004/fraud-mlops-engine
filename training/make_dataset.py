"""Split the raw Kaggle data. Everything stays RAW (unscaled); scaling lives in the model artifact."""
import pandas as pd
from sklearn.model_selection import train_test_split

from training.config import DATA_DIR, HOLDOUT_PATH, RAW_PATH, REFERENCE_PATH, STREAM_PATH, VAL_PATH


def prepare_data():
    df = pd.read_csv(RAW_PATH)
    # 60% train | 20% stream pool (simulated live traffic) | 10% validation | 10% holdout
    train, rest = train_test_split(df, test_size=0.40, random_state=42, stratify=df["Class"])
    stream, rest = train_test_split(rest, test_size=0.50, random_state=42, stratify=rest["Class"])
    val, holdout = train_test_split(rest, test_size=0.50, random_state=42, stratify=rest["Class"])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, part, path in [("train/reference", train, REFERENCE_PATH), ("stream", stream, STREAM_PATH),
                             ("validation", val, VAL_PATH), ("holdout", holdout, HOLDOUT_PATH)]:
        part.reset_index(drop=True).to_parquet(path, index=False)
        print(f"{name:16s} {part.shape}  fraud rate {part['Class'].mean():.4%}  -> {path}")


if __name__ == "__main__":
    prepare_data()
