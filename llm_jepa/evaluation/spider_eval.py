from __future__ import annotations

import sqlite3
from pathlib import Path


def execute_sql(db_path: str | Path, query: str) -> list[tuple]:
    with sqlite3.connect(str(db_path)) as connection:
        cursor = connection.cursor()
        cursor.execute(query)
        return cursor.fetchall()


def spider_execution_match(spider_root: str | Path, db_id: str, prediction: str, target: str) -> bool:
    db_file = Path(spider_root) / db_id / f"{db_id}.sqlite"
    if not db_file.exists():
        raise FileNotFoundError(f"Spider database not found: {db_file}")
    try:
        return execute_sql(db_file, prediction) == execute_sql(db_file, target)
    except sqlite3.Error:
        return False
