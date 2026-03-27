from contextlib import contextmanager
from typing import Any

from psycopg import connect
from psycopg.rows import dict_row
from psycopg import sql


@contextmanager
def get_connection(database_url: str):
    conn = connect(database_url, row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _table_identifier(table: str) -> sql.Identifier:
    return sql.Identifier(table)


def insert_row(database_url: str, table: str, data: dict) -> None:
    columns = list(data.keys())
    values = list(data.values())

    query = sql.SQL("""
        INSERT INTO {table} ({fields})
        VALUES ({placeholders})
        ON CONFLICT (uuid) DO NOTHING
    """).format(
        table=_table_identifier(table),
        fields=sql.SQL(", ").join(sql.Identifier(col) for col in columns),
        placeholders=sql.SQL(", ").join(sql.Placeholder() for _ in columns),
    )

    with get_connection(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, values)


def fetch_rows_by_uuids(database_url: str, table: str, uuids: list[str]) -> list[dict]:
    if not uuids:
        return []

    query = sql.SQL("SELECT * FROM {table} WHERE uuid = ANY(%s)").format(
        table=_table_identifier(table)
    )

    with get_connection(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(query, (uuids,))
            return list(cur.fetchall())


def fetch_rows(database_url: str, table: str, filters: dict | None = None, limit: int | None = None) -> list[dict]:
    params: list[Any] = []
    base_query = sql.SQL("SELECT * FROM {table}").format(table=_table_identifier(table))

    if filters:
        where_parts = []
        for key, value in filters.items():
            where_parts.append(sql.SQL("{} = %s").format(sql.Identifier(key)))
            params.append(value)
        base_query += sql.SQL(" WHERE ") + sql.SQL(" AND ").join(where_parts)

    if limit is not None:
        base_query += sql.SQL(" LIMIT %s")
        params.append(limit)

    with get_connection(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(base_query, params)
            return list(cur.fetchall())


def insert_decision_rows(database_url: str, rows: list[dict]) -> None:
    if not rows:
        return

    query = """
        INSERT INTO trading_bot_decision (
            timestamp, ticker, status,
            action, confidence, size_pct, time_in_force, reasoning,
            order_id, notional
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (timestamp, ticker) DO NOTHING
    """

    values = [
        (
            r.get("timestamp"),
            r.get("ticker"),
            r.get("status"),
            r.get("action"),
            r.get("confidence"),
            r.get("size_pct"),
            r.get("time_in_force"),
            r.get("reasoning"),
            r.get("order_id"),
            float(r["notional"]) if r.get("notional") not in (None, "") else None,
        )
        for r in rows
    ]

    with get_connection(database_url) as conn:
        with conn.cursor() as cur:
            cur.executemany(query, values)
