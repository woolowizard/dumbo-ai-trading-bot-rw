"""
News Pipeline: Alpha Vantage -> AI News Agent
======================================
Flusso:
  1. Recupera news da Alpha Vantage (titolo, sommario, url, fonte)
  2. Per ogni notizia News Agent produce: riassunto, segnale, rischio, alert
  3. Assembla un prompt aggregato con tutte le notizie
  4. gpt-4o-mini comprime il prompt in 5 righe (news_summary)
     pronto per essere inserito nel context block finale

Requisiti:
    pip install requests openai
"""

import json
import requests
from openai import OpenAI
from datetime import datetime
import uuid

from src.trading_bot.config import settings
from utils import db_utils

DATABASE_URL = settings.db.url

# ─────────────────────────────────────────
# STEP 1 — Recupera news da Alpha Vantage
# ─────────────────────────────────────────
def fetch_news(ticker: str, limit: int = 10) -> list[dict]:
    resp = requests.get(
        settings.api.av_base_url,
        params={
            "function": "NEWS_SENTIMENT",
            "tickers": ticker,
            "limit": limit,
            "sort": "LATEST",
            "apikey": settings.api.av_api_key,
        },
        timeout=10
    )
    resp.raise_for_status()

    articles = []
    for item in resp.json().get("feed", []):
        articles.append({
            "titolo": item.get("title", ""),
            "sommario": item.get("summary", ""),
            "fonte": item.get("source", ""),
            "url": item.get("url", ""),
            "data": item.get("time_published", ""),
            "sentiment_av": item.get("overall_sentiment_label", "Neutral"),
        })
    return articles[:limit]


def check_if_exist(articles: list[dict], existing_rows: list[dict]) -> tuple[list[dict], list[dict]]:
    existing_uuids = {row["uuid"] for row in existing_rows}
    db_by_uuid = {row["uuid"]: row for row in existing_rows}

    new_articles: list[dict] = []
    cached_analyzed: list[dict] = []

    for art in articles:
        art_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, art["titolo"]))

        if art_uuid in existing_uuids:
            db_row = db_by_uuid[art_uuid]
            cached_analyzed.append({
                **art,
                "uuid": art_uuid,
                "riassunto": db_row.get("summary", ""),
                "affidabilita_fonte": _parse_affidabilita(db_row.get("fonte", "")),
                **_parse_segnale_rischio(db_row.get("segnale", "")),
                "motivazione": db_row.get("motivo", ""),
                "alert": db_row.get("alert", "No") == "Si",
                "_from_cache": True,
            })
        else:
            new_articles.append(art)

    return new_articles, cached_analyzed


def _parse_affidabilita(fonte_str: str) -> str:
    if "affidabilità:" in fonte_str:
        return fonte_str.split("affidabilità:")[-1].strip().rstrip(")")
    return "?"


def _parse_segnale_rischio(segnale_str: str) -> dict:
    try:
        parts = segnale_str.split("|")
        segnale = parts[0].strip()
        rischio = int(parts[1].strip().replace("Rischio:", "").replace("/10", "").strip())
        return {"segnale": segnale, "rischio": rischio}
    except Exception:
        return {"segnale": "?", "rischio": 5}


def analyze_article(client: OpenAI, ticker: str, art: dict) -> dict:
    prompt = f"""Sei un analista finanziario. Analizza questa notizia sul titolo {ticker}.

TITOLO  : {art['titolo']}
FONTE   : {art['fonte']}
URL     : {art['url']}
SOMMARIO: {art['sommario']}
SENTIMENT Alpha Vantage: {art['sentiment_av']}

Nota: usa URL e fonte per valutare l'affidabilità (Reuters/Bloomberg = alta,
blog sconosciuto = bassa). Non scaricare il contenuto, ragiona solo dai dati forniti.

Rispondi SOLO con un oggetto JSON, nessun testo extra:
{{
  "riassunto": "2-3 righe in italiano: cosa è successo e perché importa per {ticker}",
  "segnale": "BUY | SELL | HOLD | WATCH",
  "rischio": <1-10, dove 10 è massimo rischio>,
  "alert": <true se notizia critica o mossa di mercato attesa>,
  "affidabilita_fonte": "alta | media | bassa",
  "motivazione": "una riga che spiega il segnale scelto"
}}"""

    message = client.chat.completions.create(
        model=settings.forecast.seeking_model,
        messages=[{"role": "user", "content": prompt}]
    )

    raw = message.choices[0].message.content.strip()

    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return {**art, **json.loads(raw)}


def build_final_prompt(ticker: str, analyzed: list[dict]) -> str:
    alerts = [a for a in analyzed if a.get("alert")]
    buy = [a for a in analyzed if a.get("segnale") == "BUY"]
    sell = [a for a in analyzed if a.get("segnale") == "SELL"]
    avg_risk = sum(a.get("rischio", 5) for a in analyzed) / len(analyzed)

    notizie_str = ""
    for i, a in enumerate(analyzed, 1):
        cache_tag = " [cache]" if a.get("_from_cache") else ""
        notizie_str += f"""
[{i}]{cache_tag} {a['titolo']}
  Fonte     : {a['fonte']} (affidabilità: {a.get('affidabilita_fonte','?')})
  Link      : {a['url']}
  Riassunto : {a.get('riassunto', '')}
  Segnale   : {a.get('segnale','?')}  |  Rischio: {a.get('rischio','?')}/10
  Motivo    : {a.get('motivazione','')}
  Alert     : {'Si' if a.get('alert') else 'No'}
"""

    return f"""=== ANALISI NEWS: {ticker} ===
Data analisi : {datetime.now().strftime('%d/%m/%Y %H:%M')}
Notizie      : {len(analyzed)}
Rischio medio: {avg_risk:.1f}/10
BUY signals  : {len(buy)}  |  SELL signals: {len(sell)}
Alert critici: {len(alerts)}

--- NOTIZIE ANALIZZATE ---
{notizie_str}
--- FINE CONTESTO ---

Sulla base delle notizie sopra, fornisci una valutazione complessiva su {ticker}:
- Sentiment generale del mercato
- Raccomandazione operativa basata solo sulle NEWS, che andrà poi integrata con dati quantitativi.
- Principali rischi da monitorare
"""


def generate_row(a: dict) -> dict:
    news_uuid = str(uuid.uuid5(uuid.NAMESPACE_URL, a["titolo"]))
    row = {
        "uuid": news_uuid,
        "fonte": f"{a['fonte']} (affidabilità: {a.get('affidabilita_fonte','?')})",
        "link": a["url"],
        "summary": a.get("riassunto", ""),
        "segnale": f"{a.get('segnale','?')}  |  Rischio: {a.get('rischio','?')}/10",
        "motivo": a.get("motivazione", ""),
        "alert": "Si" if a.get("alert") else "No",
    }
    return row


def summarize_news_prompt(client: OpenAI, news_prompt: str) -> str:
    message = client.chat.completions.create(
        model=settings.forecast.seeking_model,
        messages=[{
            "role": "user",
            "content": f"""Riassumi questo report di notizie finanziarie in massimo 5 righe.
Includi solo: sentiment generale, segnali forti (BUY/SELL), alert critici, rischi chiave.
Sii telegrafico, niente frasi di contorno. Niente elenchi puntati, solo testo compatto.
Il prompt che generi lo dovrò mandare in pasto ad un agente che si occuperà di trading.
{news_prompt}"""
        }]
    )
    return message.choices[0].message.content.strip()


def run_pipeline(ticker: str, limit: int | None = None) -> dict:
    client = OpenAI(api_key=settings.api.openai_api_key)

    if limit is None:
        limit = settings.news.news_limit

    print(f"\n Recupero news per {ticker}...")
    articles = fetch_news(ticker, limit)
    print(f"   {len(articles)} articoli trovati")

    all_uuids = [str(uuid.uuid5(uuid.NAMESPACE_URL, a["titolo"])) for a in articles]
    existing_rows = db_utils.fetch_rows_by_uuids(DATABASE_URL, "trading_bot_news", all_uuids)

    new_articles, cached_analyzed = check_if_exist(articles, existing_rows)
    print(f"   {len(cached_analyzed)} in cache  |  {len(new_articles)} nuovi da analizzare")

    fresh_analyzed: list[dict] = []
    if new_articles:
        print("\n Analisi con news agent...")
        for i, art in enumerate(new_articles, 1):
            print(f"   [{i}/{len(new_articles)}] {art['titolo'][:60]}...")
            try:
                result = analyze_article(client, ticker, art)
                fresh_analyzed.append(result)
            except Exception as e:
                print(f"   Errore articolo {i}: {e}")
    else:
        print("\n Tutti gli articoli già in cache, nessuna chiamata AI necessaria.")

    if fresh_analyzed:
        print(f"\n Scrittura {len(fresh_analyzed)} nuovi articoli sul DB...")
        for art in fresh_analyzed:
            row = generate_row(art)
            db_utils.insert_row(database_url=DATABASE_URL, table="trading_bot_news", data=row)

    analyzed = fresh_analyzed + cached_analyzed

    if not analyzed:
        print(" Nessun articolo disponibile per il prompt.")
        return {"full_prompt": "", "news_summary": ""}

    full_prompt = build_final_prompt(ticker, analyzed)

    print("\n Compressione...")
    news_summary = summarize_news_prompt(client, full_prompt)
    print(f"\n NEWS SUMMARY:\n{news_summary}")

    return {
        "full_prompt": full_prompt,
        "news_summary": news_summary,
    }


if __name__ == "__main__":
    result = run_pipeline(settings.market.ticker, limit=settings.news.news_limit)
