from datetime import datetime

from src.trading_bot.config import settings
from src.trading_bot.trader import run_agent
from src.trading_bot.context_block_builder import build_context_block

from utils.db_init import create_db
from utils.db_utils import insert_decision_rows


def build_decision_row(timestamp: str, ticker: str, agent_result: dict | None) -> dict | None:
    if not agent_result:
        return None

    execution_result = agent_result.get("execution_result") or {}
    decision = execution_result.get("decision") or agent_result.get("decision") or {}

    if not execution_result and not decision:
        return None

    notional = execution_result.get("notional")
    try:
        notional_value = float(notional) if notional not in (None, "") else None
    except (TypeError, ValueError):
        notional_value = None

    return {
        "timestamp": timestamp,
        "ticker": ticker,
        "status": execution_result.get("status"),
        "action": decision.get("action"),
        "confidence": decision.get("confidence"),
        "size_pct": decision.get("size_pct"),
        "time_in_force": decision.get("time_in_force"),
        "reasoning": decision.get("reasoning"),
        "order_id": execution_result.get("order_id"),
        "notional": notional_value,
    }


def run():
    
    create_db(settings.db.url)

    ticker = settings.market.ticker
    database_url = settings.db.url

    try:
        context_block = build_context_block(
            market_config=settings.market,
            forecast_config=settings.forecast,
            log_config=settings.logs,
            news_config=settings.news,
        )

        agent_result = run_agent(ticker, context_block)
        run_timestamp = datetime.now().isoformat()
        row = build_decision_row(run_timestamp, ticker, agent_result)
        
        print('Decision:', row)
        
        if row:
            insert_decision_rows(database_url, [row])

        return agent_result

    except Exception as e:
        
        # Qui va aggiunta la gestione dell'eccezione
        print("Exception:", str(e))

        return None
