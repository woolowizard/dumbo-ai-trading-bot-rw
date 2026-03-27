from psycopg import connect


def create_db(database_url: str) -> None:
    with connect(database_url) as conn:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_bot_news (
                    uuid TEXT PRIMARY KEY,
                    fonte TEXT DEFAULT NULL,
                    link TEXT DEFAULT NULL,
                    summary TEXT DEFAULT NULL,
                    segnale TEXT DEFAULT NULL,
                    motivo TEXT DEFAULT NULL,
                    alert TEXT DEFAULT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
                )
                """
            )

            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trading_bot_decision (
                    id BIGSERIAL PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    status TEXT DEFAULT NULL,
                    action TEXT DEFAULT NULL,
                    confidence DOUBLE PRECISION DEFAULT NULL,
                    size_pct DOUBLE PRECISION DEFAULT NULL,
                    time_in_force TEXT DEFAULT NULL,
                    reasoning TEXT DEFAULT NULL,
                    order_id TEXT DEFAULT NULL,
                    notional DOUBLE PRECISION DEFAULT NULL,
                    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(timestamp, ticker)
                )
                """
            )
        conn.commit()

    print("Database PostgreSQL inizializzato correttamente.")
