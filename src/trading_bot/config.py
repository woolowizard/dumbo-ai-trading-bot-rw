# src/trading_bot/config.py

import os
from dataclasses import dataclass
from urllib.parse import quote_plus
from dotenv import load_dotenv

load_dotenv()


def _get_env(name: str, default: str | None = None, required: bool = False) -> str:
    value = os.getenv(name, default)
    if required and (value is None or value == ""):
        raise ValueError(f"Manca la variabile d'ambiente obbligatoria: {name}")
    return value


@dataclass(frozen=True)
class MarketConfig:
    ticker: str = os.getenv("MARKET_TICKER", "AAPL")
    interval: str = os.getenv("MARKET_INTERVAL", "15m")
    period: str = os.getenv("MARKET_PERIOD", "59d")


@dataclass(frozen=True)
class ForecastConfig:
    n_lags: int = int(os.getenv("FORECAST_N_LAGS", "16"))
    forecast_horizon: int = int(os.getenv("FORECAST_HORIZON", "1"))
    cv_boost_factor: float = float(os.getenv("FORECAST_CV_BOOST_FACTOR", "1.1"))
    seeking_model: str = os.getenv("OPENAI_MODEL", "gpt-5-nano")


@dataclass(frozen=True)
class LogConfig:
    log_prompt_file: str = os.getenv("LOG_PROMPT_FILE", "data/prompt_log.txt")
    log_decision_file: str = os.getenv("LOG_DECISION_FILE", "data/decision_log.txt")
    log_file: str = os.getenv("LOG_FILE", "data/predictions_log.txt")
    log_dir: str = os.getenv("LOG_DIR", "data")


@dataclass(frozen=True)
class APIConfig:
    openai_api_key: str = _get_env("OPENAI_API_KEY", required=True)
    api_key: str = _get_env("API_KEY", required=True)
    api_secret: str = _get_env("API_SECRET", required=True)
    av_api_key: str = _get_env("AV_API_KEY", required=True)
    av_base_url: str = "https://www.alphavantage.co/query"


@dataclass(frozen=True)
class NewsConfig:
    news_limit: int = int(os.getenv("NEWS_LIMIT", "1"))


@dataclass(frozen=True)
class DBConfig:
    database_url: str | None = os.getenv("DATABASE_URL")
    host: str = os.getenv("DB_HOST", os.getenv("PGHOST", ""))
    port: int = int(os.getenv("DB_PORT", os.getenv("PGPORT", "5432")))
    dbname: str = os.getenv("DB_NAME", os.getenv("PGDATABASE", ""))
    user: str = os.getenv("DB_USER", os.getenv("PGUSER", ""))
    password: str = os.getenv("DB_PASSWORD", os.getenv("PGPASSWORD", ""))
    sslmode: str = os.getenv("DB_SSLMODE", "require")

    @property
    def url(self) -> str:
        if self.database_url:
            return self.database_url

        required = {
            "DB_HOST/PGHOST": self.host,
            "DB_NAME/PGDATABASE": self.dbname,
            "DB_USER/PGUSER": self.user,
            "DB_PASSWORD/PGPASSWORD": self.password,
        }
        missing = [name for name, value in required.items() if not value]
        if missing:
            raise ValueError(
                "Configurazione DB incompleta. Imposta DATABASE_URL oppure: " + ", ".join(missing)
            )

        safe_user = quote_plus(self.user)
        safe_password = quote_plus(self.password)
        safe_dbname = quote_plus(self.dbname)
        return (
            f"postgresql://{safe_user}:{safe_password}@{self.host}:{self.port}/{safe_dbname}"
            f"?sslmode={self.sslmode}"
        )


@dataclass(frozen=True)
class Settings:
    market: MarketConfig = MarketConfig()
    forecast: ForecastConfig = ForecastConfig()
    logs: LogConfig = LogConfig()
    api: APIConfig = APIConfig()
    news: NewsConfig = NewsConfig()
    db: DBConfig = DBConfig()


settings = Settings()
