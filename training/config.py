import os
from pathlib import Path

from xgboost import XGBClassifier

DATA_DIR = Path(os.getenv("DATA_DIR", "data/processed"))
RAW_PATH = Path(os.getenv("RAW_DATA_PATH", "data/raw/creditcard.csv"))
REFERENCE_PATH = DATA_DIR / "reference_baseline.parquet"   # train split, RAW features + Class
VAL_PATH = DATA_DIR / "validation.parquet"                 # used ONLY to tune the threshold
HOLDOUT_PATH = DATA_DIR / "holdout_test.parquet"           # used ONLY for the promotion gate
STREAM_PATH = DATA_DIR / "stream_pool.parquet"             # used ONLY by the simulator


def build_model(scale_pos_weight: float, n_estimators: int = 300, max_depth: int = 5) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.05,
        scale_pos_weight=scale_pos_weight, eval_metric="aucpr", tree_method="hist",
        random_state=42, n_jobs=-1,
    )
