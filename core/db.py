"""Database layer. Postgres in the cloud (DATABASE_URL), SQLite fallback for local dev."""
import json
import os
from datetime import datetime, timedelta

import pandas as pd
from sqlalchemy import (Column, DateTime, Float, Integer, MetaData, String, Table,
                        Text, create_engine, func, select, update)

_metadata = MetaData()
inference_logs = Table(
    "inference_logs", _metadata,
    Column("txn_id", String(64), primary_key=True),
    Column("timestamp", DateTime, nullable=False),
    Column("features_json", Text, nullable=False),
    Column("risk_score", Float, nullable=False),
    Column("is_flagged", Integer, nullable=False),
    Column("model_version", String(32)),
    Column("actual_label", Integer),
    Column("feedback_timestamp", DateTime),
)

_engine = None


def get_engine():
    global _engine
    if _engine is None:
        url = os.getenv("DATABASE_URL", "sqlite:///fraud_logs.db")
        if url.startswith("postgres://"):
            url = url.replace("postgres://", "postgresql://", 1)
        kwargs = {"pool_pre_ping": True}
        if not url.startswith("sqlite"):
            kwargs.update(pool_size=2, max_overflow=2, pool_recycle=300)
        _engine = create_engine(url, **kwargs)
    return _engine


def reset_engine():
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = None


def init_db():
    _metadata.create_all(get_engine())


def log_prediction(txn_id: str, features: dict, risk_score: float,
                   is_flagged: bool, model_version: str) -> None:
    with get_engine().begin() as conn:
        conn.execute(inference_logs.insert().values(
            txn_id=txn_id, timestamp=datetime.utcnow(),
            features_json=json.dumps(features), risk_score=float(risk_score),
            is_flagged=int(is_flagged), model_version=model_version))


def log_feedback(txn_id: str, actual_label: int) -> bool:
    """Attach ground truth. Returns False when the txn_id does not exist."""
    with get_engine().begin() as conn:
        res = conn.execute(update(inference_logs)
                           .where(inference_logs.c.txn_id == txn_id)
                           .values(actual_label=int(actual_label),
                                   feedback_timestamp=datetime.utcnow()))
        return res.rowcount > 0


def fetch_logs(labeled_only: bool = False, since_hours: float | None = None,
               limit: int | None = None) -> pd.DataFrame:
    stmt = select(inference_logs).order_by(inference_logs.c.timestamp.desc())
    if labeled_only:
        stmt = stmt.where(inference_logs.c.actual_label.is_not(None))
    if since_hours is not None:
        stmt = stmt.where(inference_logs.c.timestamp >= datetime.utcnow() - timedelta(hours=since_hours))
    if limit:
        stmt = stmt.limit(limit)
    with get_engine().connect() as conn:
        return pd.read_sql(stmt, conn)


def count_labeled() -> int:
    stmt = select(func.count()).select_from(inference_logs).where(inference_logs.c.actual_label.is_not(None))
    with get_engine().connect() as conn:
        return int(conn.execute(stmt).scalar() or 0)


def expand_features(df_logs: pd.DataFrame) -> pd.DataFrame:
    """Unpack features_json into one column per feature (raw values, same index as df_logs)."""
    if df_logs.empty:
        return pd.DataFrame()
    return pd.DataFrame([json.loads(s) for s in df_logs["features_json"]], index=df_logs.index)


if __name__ == "__main__":
    init_db()
    print("Database initialised.")
