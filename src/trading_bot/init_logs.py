# ─────────────────────────────────────────────
# Context block builder
# ─────────────────────────────────────────────
import os
from datetime import datetime


def init_logs(market_config, log_config):
    ticker = market_config.ticker

    log_file = log_config.log_file
    log_prompt_file = log_config.log_prompt_file
    log_decision_file = log_config.log_decision_file
    log_folder_name = log_config.log_dir

    # Crea la cartella se non esiste
    os.makedirs(log_folder_name, exist_ok=True)

    # Inizializza i file che verranno scritti quando il bot è in funzione
    if not os.path.exists(log_file):
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"# Stock Forecaster — {ticker} | Avviato {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write("# Formato: timestamp | candle | last_close | predicted | change%\n")
            f.write("-" * 90 + "\n")

    if not os.path.exists(log_prompt_file):
        with open(log_prompt_file, "w", encoding="utf-8") as f:
            f.write(f"# Context block prompt for — {ticker} | Avviato {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write("-" * 90 + "\n")

    if not os.path.exists(log_decision_file):
        with open(log_decision_file, "w", encoding="utf-8") as f:
            f.write(f"# Decision log for — {ticker} | Avviato {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write("-" * 90 + "\n")

    print("log inizializzati")