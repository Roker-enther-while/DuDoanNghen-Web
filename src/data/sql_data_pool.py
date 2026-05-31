from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Iterable

import pandas as pd


COMMON_SCHEMA = [
    "timestamp",
    "source_name",
    "machine_id",
    "service_id",
    "cpu_usage",
    "memory_usage",
    "disk_io",
    "network_in",
    "network_out",
    "request_rate",
    "throughput",
    "response_time",
    "error_rate",
    "congestion_label",
    "is_synthetic",
    "is_noisy",
    "time_index",
]

REQUIRED_TABLES = [
    "raw_old_bitbrains",
    "old_bitbrains_train80",
    "old_bitbrains_holdout20",
    "raw_external_logs",
    "harmonized_external_logs",
    "synthetic_noisy_logs",
    "train_pool",
    "validation_pool",
    "test_pool_optional",
    "source_inventory",
    "data_quality_report",
]


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    return conn


def initialize_database(db_path: str | Path) -> None:
    with connect(db_path) as conn:
        for table in REQUIRED_TABLES:
            conn.execute(f'CREATE TABLE IF NOT EXISTS "{table}" (placeholder INTEGER)')
        conn.commit()


def replace_table(db_path: str | Path, table: str, df: pd.DataFrame) -> None:
    with connect(db_path) as conn:
        df.to_sql(table, conn, if_exists="replace", index=False)
        if "timestamp" in df.columns:
            try:
                conn.execute(f'CREATE INDEX IF NOT EXISTS idx_{table}_time ON "{table}" (timestamp, time_index)')
            except sqlite3.OperationalError:
                pass
        conn.commit()


def append_table(db_path: str | Path, table: str, chunks: Iterable[pd.DataFrame]) -> int:
    total = 0
    with connect(db_path) as conn:
        first = True
        for chunk in chunks:
            chunk.to_sql(table, conn, if_exists="replace" if first else "append", index=False)
            total += len(chunk)
            first = False
        conn.commit()
    return total


def read_ordered(db_path: str | Path, table: str, limit: int | None = None) -> pd.DataFrame:
    limit_sql = f" LIMIT {int(limit)}" if limit else ""
    query = f'SELECT * FROM "{table}" ORDER BY timestamp ASC, time_index ASC{limit_sql}'
    with connect(db_path) as conn:
        return pd.read_sql_query(query, conn)


def count_rows(db_path: str | Path, table: str) -> int:
    with connect(db_path) as conn:
        row = conn.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()
    return int(row[0])

