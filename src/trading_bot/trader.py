import os
import json
from decimal import Decimal
from typing import Any, Dict, Optional

from openai import OpenAI
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce

# TODO: Vedere se funziona rimuovendo questi

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
API_KEY = os.environ["API_KEY"]
API_SECRET = os.environ["API_SECRET"]

if not OPENAI_API_KEY:
    raise ValueError("Manca OPENAI_API_KEY")

if not API_KEY or not API_KEY:
    raise ValueError("Mancano API_KEY / API_SECRET")

llm_client = OpenAI(api_key=OPENAI_API_KEY)
alpaca = TradingClient(API_KEY, API_SECRET, paper=True)

ALLOWED_ACTIONS = {"buy", "sell", "hold"}
ALLOWED_TIF = {"day", "gtc"}

def normalize_model_decision(raw: Dict[str, Any]) -> Dict[str, Any]:
    action = str(raw.get("action", "hold")).lower().strip()
    if action not in ALLOWED_ACTIONS:
        action = "hold"

    try:
        confidence = float(raw.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    try:
        size_pct = float(raw.get("size_pct", 0.0))
    except Exception:
        size_pct = 0.0
    size_pct = max(0.0, min(1.0, size_pct))

    tif = str(raw.get("time_in_force", "day")).lower().strip()
    if tif not in ALLOWED_TIF:
        tif = "day"

    reasoning = str(raw.get("reasoning", "")).strip()

    return {
        "action": action,
        "confidence": confidence,
        "size_pct": size_pct,
        "time_in_force": tif,
        "reasoning": reasoning,
    }


def get_position_state(symbol: str) -> dict | None:
    """
    Funzione che ritorna lo stato delle posizioni aperte in modo
    da dare all'agente il contesto delle operazioni precedenti. È
    un ulteriore metrica di valutazione su cui si basa l'agente.
    """
    try:
        p = alpaca.get_open_position(symbol)
        return {
            "symbol": p.symbol,
            "qty": str(p.qty),
            "side": str(p.side),
            "market_value": str(p.market_value),
            "avg_entry_price": str(p.avg_entry_price),
            "unrealized_pl": str(p.unrealized_pl),
        }
    except Exception:
        return None


def enrich_context_block(symbol: str, context_block: dict) -> dict:
    """
    Funzione che amplica il context_block con altre informazioni quali:
    *) symbol: ticker;
    *) account_state: stato dell'account definito dal potere di acquisto, cash ed equity;
    *) position_state: contesto del trading fatto fino ad ora;
    """
    account = alpaca.get_account()

    enriched = {
        **context_block,
        "symbol": symbol,
        "account_state": {
            "buying_power": str(account.buying_power),
            "cash": str(account.cash),
            "equity": str(account.equity),
        },
        "position_state": get_position_state(symbol),
    }
    return enriched


def call_llm_for_decision(context_block: Dict[str, Any]) -> Dict[str, Any]:
    response = llm_client.responses.create(
        model="gpt-5-nano",
        input=[
            {
                "role": "system",
                "content": """
You are an autonomous trading decision engine.

You receive a context block containing:
- news_summary
- forecast
- quantitative_indicators
- symbol
- account_state
- position_state

Decide one action: buy, sell, or hold.

Important:
- The trading decision must come from the context.
- Do not apply hard-coded trading rules.
- Use discretionary judgment from the supplied information.
- Return only valid JSON.

Required JSON schema:
{
  "action": "buy" | "sell" | "hold",
  "confidence": 0.0,
  "size_pct": 0.0,
  "time_in_force": "day" | "gtc",
  "reasoning": "brief explanation"
}
"""
            },
            {
                "role": "user",
                "content": json.dumps(context_block, ensure_ascii=False)
            }
        ]
    )

    raw_text = response.output_text.strip()

    try:
        parsed = json.loads(raw_text)
    except Exception:
        parsed = {
            "action": "hold",
            "confidence": 0.0,
            "size_pct": 0.0,
            "time_in_force": "day",
            "reasoning": "Invalid JSON returned by model"
        }

    return normalize_model_decision(parsed)


def execute_decision(symbol: str, decision: Dict[str, Any]) -> Dict[str, Any]:
    if decision["action"] == "hold":
        return {"status": "no_order", "decision": decision}

    account = alpaca.get_account()
    buying_power = Decimal(str(account.buying_power))
    notional = (buying_power * Decimal(str(decision["size_pct"]))).quantize(Decimal("0.01"))

    if notional <= Decimal("1.00"):
        return {
            "status": "no_order",
            "reason": "notional_too_small",
            "decision": decision
        }

    side = OrderSide.BUY if decision["action"] == "buy" else OrderSide.SELL
    tif = TimeInForce.DAY if decision["time_in_force"] == "day" else TimeInForce.GTC

    order_data = MarketOrderRequest(
        symbol=symbol,
        notional=float(notional),
        side=side,
        time_in_force=tif,
    )

    order = alpaca.submit_order(order_data=order_data)

    return {
        "status": "submitted",
        "symbol": symbol,
        "decision": decision,
        "order_id": str(order.id),
        "notional": str(notional),
    }


def run_agent(symbol: str, context_block: dict) -> dict:
    enriched_context = enrich_context_block(symbol, context_block)
    decision = call_llm_for_decision(enriched_context)
    execution_result = execute_decision(symbol, decision)

    return {
        "context_block": enriched_context,
        "decision": decision,
        "execution_result": execution_result,
    }
