from typing import List
import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

warnings.filterwarnings("ignore")

from dataExtractor import bloombergFetch, BloombergData, PortfolioPosition
from portfolioEngine import PortfolioEngine
from VaRengine import VaREngine
from kelly import KellyCriterion


# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURE YOUR PORTFOLIO HERE
# ═══════════════════════════════════════════════════════════════════════════════
PORTFOLIO: List[PortfolioPosition] = [
    PortfolioPosition(ticker="AAPL US Equity",  quantity=150,  cost_basis=172.50, asset_class="Equity"),
    PortfolioPosition(ticker="MSFT US Equity",  quantity=80,   cost_basis=390.00, asset_class="Equity"),
    PortfolioPosition(ticker="GOOGL US Equity", quantity=60,   cost_basis=135.00, asset_class="Equity"),
    PortfolioPosition(ticker="AMZN US Equity",  quantity=100,  cost_basis=178.00, asset_class="Equity"),
    PortfolioPosition(ticker="NVDA US Equity",  quantity=50,   cost_basis=480.00, asset_class="Equity"),
]

# Risk parameters
CONFIDENCE     = 0.95
HORIZON_DAYS   = 1
RF_ANNUAL      = 0.0525       # UK base rate approximation
VAR_LOOKBACK   = 504          # 2 years of trading days
KELLY_LOOKBACK = 252          # 1 year


# ═══════════════════════════════════════════════════════════════════════════════
def load_data() -> BloombergData:
    tickers = [p.ticker for p in PORTFOLIO]
    start = date.today() - timedelta(days=760)
    
    return bloombergFetch(tickers, start)


def run_var(data: BloombergData, weights: Series, total_value: float):
    print("\n" + "═" * 60)
    print("  VALUE-AT-RISK REPORT")
    print("═" * 60)

    engine = VaREngine(data.returns, weights, total_value)

    # ── Full report table
    print(f"\n  Confidence: {CONFIDENCE:.0%}  |  Horizon: {HORIZON_DAYS}d  |  Lookback: {VAR_LOOKBACK}d")
    print()
    print(engine.full_report(CONFIDENCE, HORIZON_DAYS).to_string())

    # ── Individual method detail
    print()
    for method_fn in [engine.historical, engine.parametric, engine.monte_carlo]:
        r = method_fn(CONFIDENCE, HORIZON_DAYS)
        print(f"  {r}")

    # ── Stress tests
    print("\n── Stress Scenarios ─────────────────────────────────────")
    print(engine.stress_test().to_string())


def run_portfolio(engine: PortfolioEngine):
    engine.printOutput()


def run_kelly(data: BloombergData, tickers: List[str]):
    print("\n" + "═" * 60)
    print("  KELLY CRITERION REPORT")
    print("═" * 60)

    kc = KellyCriterion(data.returns, RF_ANNUAL)

    # Per-ticker continuous Kelly
    print("\n── Per-Ticker Kelly Summary ─────")
    print(kc.summary_table(tickers, KELLY_LOOKBACK).to_string())

    # Multi-asset Kelly
    print("\n── Multi-Asset Optimal Kelly Allocation ────")
    mak = kc.multi_asset(tickers, lookback=KELLY_LOOKBACK, allow_short=False, max_leverage=1.0)
    print(mak)

    # Individual discrete Kelly examples
    print("\n── Discrete Kelly (data-driven win/loss) ────")
    for t in tickers[:3]:
        result = kc.discrete(t)
        print(result)
        print()


def run_fundamentals(data: BloombergData):
    if data.fundamentals.empty:
        return
    print("\n" + "═" * 60)
    print("  FUNDAMENTALS SNAPSHOT")
    print("═" * 60)
    fd = data.fundamentals.copy()
    # Rename for readability
    fd.columns = [c.replace("_", " ").title() for c in fd.columns]
    print(fd.round(2).to_string())


def run_rolling(engine: PortfolioEngine):
    roll = engine.rolling_metrics(window=63)
    latest = roll.iloc[-1]
    print("\n── Rolling Metrics (last 63d window) ─────")
    for k, v in latest.items():
        print(f"  {k:<30}: {v:+.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
def main():
    data: BloombergData = load_data()

    # ── Build portfolio engine
    port_engine = PortfolioEngine(PORTFOLIO, data, RF_ANNUAL)
    tickers = [p.ticker for p in PORTFOLIO]

    # ── Run all modules
    run_portfolio(port_engine)
    run_var(data, port_engine.weights, port_engine.total_market_value)
    run_kelly(data, tickers)
    run_fundamentals(data)
    run_rolling(port_engine)

    print("\n" + "═" * 60)
    print("  DONE")
    print("═" * 60 + "\n")

main()
