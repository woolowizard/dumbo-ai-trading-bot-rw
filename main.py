# ============================================================
#  Script di avvio
# ============================================================

from src.trading_bot.main_loop import run
from src.trading_bot.config import settings
from alpaca.trading.client import TradingClient

if __name__ == '__main__':
    
    client = TradingClient(settings.api.api_key, settings.api.api_secret)
    clock = client.get_clock()
    
    if clock.is_open:
        run()
    
    