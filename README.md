# 🤖 AI Trading Bot

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![LightGBM](https://img.shields.io/badge/LightGBM-4.0+-brightgreen.svg)](https://lightgbm.readthedocs.io/)
[![Alpaca](https://img.shields.io/badge/Alpaca-Paper%20Trading-yellow.svg)](https://alpaca.markets/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--5--nano-74aa9c.svg)](https://openai.com/)

*Un sistema di trading algoritmico basato su Machine Learning e LLM per decisioni autonome sui mercati finanziari*

</div>

---

## 📋 Panoramica

Questo progetto implementa un **trading bot autonomo** che combina:
- **Predizione ML** con LightGBM per forecasting dei prezzi
- **Analisi del sentiment** delle news tramite Alpha Vantage e GPT
- **Decisioni AI** tramite LLM (GPT-5-nano) che integra dati quantitativi e qualitativi
- **Esecuzione ordini** in tempo reale su Alpaca (paper trading)

---

## 🏗️ Architettura

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Alpha Vantage │────▶│   News Agent    │────▶│   PostgreSQL    │
│   (News API)    │     │   (GPT-4o-mini) │     │   (Cache News)  │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                          │
┌─────────────────┐     ┌─────────────────┐              │
│   yFinance      │────▶│  LightGBM       │◄─────────────┘
│   (Dati Prezzi) │     │  Forecaster     │
└─────────────────┘     └─────────────────┘
                                │
                                ▼
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Alpaca        │◄────│  Trading Agent  │◄────│  Context Block  │
│   (Esecuzione)  │     │  (GPT-5-nano)   │     │  Builder        │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

---

## 🚀 Stack Tecnologico

| Componente | Tecnologia | Scopo |
|------------|-----------|-------|
| ![LightGBM](https://kimi-web-img.moonshot.cn/img/miro.medium.com/32169388d773c82640e27b932ea38b3e2ad51503) | **LightGBM** | Modello di regressione per predizione prezzi |
| ![Alpha Vantage](https://kimi-web-img.moonshot.cn/img/res.cloudinary.com/0e48d763026816cac9b0a6f64ac1463aa3d14fba.jpg) | **Alpha Vantage** | News e sentiment analysis |
| ![Alpaca](https://kimi-web-img.moonshot.cn/img/alpaca.markets/b73b1794bb6b630014f422ef4c79f83ebea24669.png) | **Alpaca Markets** | Paper trading e esecuzione ordini |
| ![OpenAI](https://kimi-web-img.moonshot.cn/img/us1.discourse-cdn.com/1df5db2d844ccdb11f31a89117d039fd3147dbdf.png) | **OpenAI GPT** | Analisi news e decisioni trading |

---

## ⚙️ Configurazione

### Variabili d'Ambiente (.env)

```bash
# API Keys
OPENAI_API_KEY=sk-...
API_KEY=AK...
API_SECRET=PK...
AV_API_KEY=...

# Configurazione Mercato
MARKET_TICKER=AAPL
MARKET_INTERVAL=15m
MARKET_PERIOD=59d

# Configurazione Forecast
FORECAST_N_LAGS=16
FORECAST_HORIZON=1
FORECAST_CV_BOOST_FACTOR=1.1

# Database PostgreSQL
DATABASE_URL=postgresql://user:pass@host:5432/dbname
# oppure
DB_HOST=localhost
DB_PORT=5432
DB_NAME=trading_bot
DB_USER=postgres
DB_PASSWORD=secret

# News
NEWS_LIMIT=3
```

---

## 📦 Installazione

```bash
# Clona il repository
git clone https://github.com/tuo-username/ai-trading-bot.git
cd ai-trading-bot

# Crea virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installa dipendenze
pip install -r requirements.txt

# Inizializza il database
python -c "from utils.db_init import create_db; from src.trading_bot.config import settings; create_db(settings.db.url)"

# Esegui il bot
python src/trading_bot/main_loop.py
```

### Requirements

```txt
lightgbm>=4.0.0
yfinance>=0.2.0
pandas>=2.0.0
numpy>=1.24.0
ta>=0.10.0
openai>=1.0.0
alpaca-py>=0.15.0
psycopg>=3.1.0
requests>=2.31.0
python-dotenv>=1.0.0
scikit-learn>=1.3.0
```

---

## 🧠 Moduli Principali

### 1. `forecast.py` - Predizione Prezzi
- Feature engineering avanzato (RSI, MACD, Bollinger Bands, ATR)
- Time Series Cross-Validation
- Modello LightGBM con early stopping

```python
from src.trading_bot.forecast import prediction_job, train_model

# Ottieni predizione
forecast = prediction_job(market_config, forecast_config)
# {'direction': 'up', 'expected_return': '1.23%', ...}
```

### 2. `get_news.py` - Analisi News
- Recupero news da Alpha Vantage
- Analisi AI con GPT-4o-mini (segnali BUY/SELL/HOLD/WATCH)
- Caching PostgreSQL per evitare chiamate duplicate
- Sintesi finale in 5 righe per il trading agent

### 3. `trader.py` - Decisioni & Esecuzione
- **Enrich context**: stato account, posizioni aperte
- **LLM Decision**: GPT-5-nano decide buy/sell/hold con confidence
- **Risk Management**: position sizing basato su conviction
- **Execution**: ordini market su Alpaca (paper trading)

### 4. `context_block_builder.py` - Orchestrazione
Assembla il context block completo:
- News summary
- Forecast quantitativo
- Indicatori tecnici (RSI, MACD, volume)

---

## 🗄️ Schema Database

```sql
-- Tabella News (cache)
CREATE TABLE trading_bot_news (
    uuid TEXT PRIMARY KEY,
    fonte TEXT,
    link TEXT,
    summary TEXT,
    segnale TEXT,      -- "BUY | Rischio: 3/10"
    motivo TEXT,
    alert TEXT,        -- "Si"/"No"
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- Tabella Decisioni
CREATE TABLE trading_bot_decision (
    id BIGSERIAL PRIMARY KEY,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    status TEXT,       -- submitted, no_order, error
    action TEXT,       -- buy, sell, hold
    confidence DOUBLE PRECISION,
    size_pct DOUBLE PRECISION,
    time_in_force TEXT,
    reasoning TEXT,
    order_id TEXT,
    notional DOUBLE PRECISION,
    UNIQUE(timestamp, ticker)
);
```

---

## 🔄 Flusso di Esecuzione

```python
# main_loop.py orchestration
1. build_context_block()
   ├── fetch_news() → analyze with GPT → cache in DB
   ├── prediction_job() → LightGBM forecast
   └── get_quant_index() → technical indicators

2. run_agent(symbol, context_block)
   ├── enrich_context_block() → account + positions
   ├── call_llm_for_decision() → AI decision
   └── execute_decision() → Alpaca order

3. persist_decision() → PostgreSQL
```

---

## 🛡️ Risk Management

Il bot implementa diverse protezioni:

| Regola | Implementazione |
|--------|-----------------|
| **Capital Preservation** | Priorità alla protezione del capitale vs profitto |
| **Position Sizing** | `size_pct` proporzionale alla confidence (0.0-1.0) |
| **Notional Minimum** | No ordini sotto $1.00 |
| **Confidence Threshold** | Hold automatico se segnali deboli/conflittuali |
| **Proactive Exit** | Possibilità di vendita anche senza profitto per protezione |

---

## 📝 Esempio Output

```
============================================================
[2026-04-03 13:25:00] Recupero news per — AAPL
   3 articoli trovati
   2 in cache  |  1 nuovi da analizzare

 Analisi con news agent...
   [1/1] Apple annuncia nuovo buyback...

 Compressione...
 NEWS SUMMARY:
 Sentiment positivo su AAPL per buyback record. Rischio medio (5/10). 
 Segnale: BUY con cautela. Alert: no.

============================================================
[2026-04-03 13:25:15] Avvio forecast per — AAPL
[2026-04-03 13:25:18] Download + training...
[2026-04-03 13:25:25] Modello pronto (n_estimators=850)
[2026-04-03 13:25:26] Forecast completato

============================================================
[2026-04-03 13:25:26] Calcolo indicatori quantitativi
[2026-04-03 13:25:27] Indicatori calcolati

Decision: {
    'timestamp': '2026-04-03T13:25:27',
    'ticker': 'AAPL',
    'action': 'buy',
    'confidence': 0.75,
    'size_pct': 0.15,
    'status': 'submitted',
    'order_id': 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx',
    'notional': 15000.0
}
```

---

## ⚠️ Disclaimer

> **Questo progetto è per scopi educativi e di ricerca.** Il trading comporta rischi significativi di perdita. Non utilizzare in produzione senza estensivo backtesting e validazione. L'autore non è responsabile per perdite finanziarie.

---

## 📄 Licenza

MIT License - vedi [LICENSE](LICENSE)

---

<div align="center">

</div>
