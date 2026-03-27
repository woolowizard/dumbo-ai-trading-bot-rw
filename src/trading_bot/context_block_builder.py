# ─────────────────────────────────────────────
# Context block builder
# ─────────────────────────────────────────────
from datetime import datetime

from src.trading_bot.get_news import run_pipeline
from src.trading_bot.forecast import (
    prediction_job, retrain_job,
    get_quant_index, train_model
)

def build_context_block(market_config, forecast_config, log_config, news_config) -> dict:
    context_block = {}

    print("=" * 60)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Recupero news per — {market_config.ticker}")
    news_result = run_pipeline(market_config.ticker, limit=news_config.news_limit)
    context_block["news_summary"] = news_result.get("news_summary")
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] News recuperate")

    print("=" * 60)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Avvio forecast per — {market_config.ticker}")
    context_block["forecast"] = prediction_job(market_config, forecast_config, log_config)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Forecast completato")

    print("=" * 60)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Calcolo indicatori quantitativi")
    _, _, raw = train_model(market_config, forecast_config)
    context_block["quantitative_indicators"] = get_quant_index(raw, forecast_config)
    print(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] Indicatori calcolati")

    return context_block
