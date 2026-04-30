from xbbg import blp
import pandas as pd
import numpy as np
from datetime import date, datetime
from typing import List, Tuple

def get_spot_price(ticker: str) -> float:
    data = blp.bdp(tickers=ticker, flds=["PX_LAST"])
    return float(data.iloc[0, 2])


def get_risk_free_rate(tenor: str = "US0003M Index") -> float:
    data = blp.bdp(tickers=tenor, flds=["PX_LAST"])
    rate_pct = float(data.iloc[0, 2])
    return rate_pct / 100.0


def get_dividend_yield(ticker: str) -> float:
    data = blp.bdp(tickers=ticker, flds=["DVD_YLD"])
    yield_pct = float(data.iloc[0, 2]) if data.iloc[0, 2] else 0.0
    return yield_pct / 100.0

# Option chain helpers

def get_option_chain(ticker: str) -> pd.DataFrame:
    import re
 
    raw = blp.bds(ticker, "OPT_CHAIN")

    col = raw.columns[2]
    option_tickers = raw[col].dropna().str.strip().tolist()
 
    pattern = re.compile(r"(\d{2}/\d{2}/\d{2})\s+([CP])([\d.]+)")
 
    records = []
    for opt in option_tickers:
        m = pattern.search(opt)
        if not m:
            continue
        try:
            expiry   = datetime.strptime(m.group(1), "%m/%d/%y").date()
            opt_type = m.group(2)           # 'C' or 'P'
            strike   = float(m.group(3))
            records.append(
                {"ticker": opt, "expiry": expiry,
                 "strike": strike, "option_type": opt_type}
            )
        except ValueError:
            continue
 
    chain = pd.DataFrame(records)
    chain.sort_values(["expiry", "option_type", "strike"], inplace=True)
    chain.reset_index(drop=True, inplace=True)
    return chain
 


def filter_chain(
    chain: pd.DataFrame,
    option_type: str = "C",
    min_expiry_days: int = 7,
    max_expiry_days: int = 365,
    moneyness_band: float = 0.30,
    spot: float | None = None,
) -> pd.DataFrame:
    today = date.today()
    chain = chain.copy()
    chain["days_to_expiry"] = (pd.to_datetime(chain["expiry"]) 
                               - pd.Timestamp(today)).dt.days

    mask = (chain["days_to_expiry"] >= min_expiry_days) & (
        chain["days_to_expiry"] <= max_expiry_days
    )

    if option_type != "both":
        mask &= chain["option_type"] == option_type

    if spot is not None:
        lo = spot * (1 - moneyness_band)
        hi = spot * (1 + moneyness_band)
        mask &= chain["strike"].between(lo, hi)

    return chain[mask].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Market data for options
# ---------------------------------------------------------------------------

def get_option_market_data(option_tickers: List[str]) -> pd.DataFrame:
    fields = ["PX_LAST", "PX_BID", "PX_ASK", "OPEN_INT", "VOLUME", "IVOL_MID"]
    data = blp.bdp(tickers=option_tickers, flds=fields).to_pandas()
    data = data.pivot(index="ticker", columns="field", values="value")
    data = data.rename(columns={None: "ticker", "PX_LAST": "px_last", "PX_BID": "px_bid", "PX_ASK": "px_ask", "OPEN_INT": "open_int", "VOLUME": "volume", "IVOL_MID": "ivol_mid"})
    # Compute mid price; fall back to last if bid/ask unavailable
    data["px_mid"] = (data["px_bid"] + data["px_ask"]) / 2.0
    data["px_mid"] = data["px_mid"].fillna(data["px_last"])

    return data


def get_option_expiries_and_strikes(
    chain: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    expiries = np.sort(chain["expiry"].unique())
    strikes = np.sort(chain["strike"].unique())
    return expiries, strikes


# Composite loader — single call that returns everything surface.py needs

def load_surface_inputs(
    underlying: str = "AAPL US Equity",
    option_type: str = "C",
    rate_ticker: str = "US0003M Index",
    min_expiry_days: int = 7,
    max_expiry_days: int = 365,
    moneyness_band: float = 0.30,
) -> None:
    print(f"[dataExtractor] Fetching spot for {underlying} …")
    spot = get_spot_price(underlying)

    print(f"[dataExtractor] Fetching risk-free rate ({rate_ticker}) …")
    rate = get_risk_free_rate(rate_ticker)

    print(f"[dataExtractor] Fetching dividend yield for {underlying} …")
    div_yield = get_dividend_yield(underlying)

    print(f"[dataExtractor] Fetching option chain …")
    full_chain = get_option_chain(underlying)

    print(f"[dataExtractor] Filtering chain (type={option_type}, "
          f"DTE {min_expiry_days}–{max_expiry_days}, ±{moneyness_band*100:.0f}% moneyness) …")
    chain: pd.DataFrame = filter_chain(
        full_chain,
        option_type=option_type,
        min_expiry_days=min_expiry_days,
        max_expiry_days=max_expiry_days,
        moneyness_band=moneyness_band,
        spot=spot,
    )

    print(f"[dataExtractor] Fetching market data for {len(chain)} contracts …")
    mkt = get_option_market_data(chain["ticker"].tolist())
    chain = chain.join(mkt, on="ticker")

    # Drop contracts with no tradeable price
    chain = chain[chain["px_mid"].notna() & (chain["px_mid"] > 0)].reset_index(drop=True)

    meta = pd.DataFrame([{
        "underlying": underlying,
        "spot": spot,
        "rate": rate,
        "div_yield": div_yield
    }])


    from pathlib import Path
    BASE_DIR = Path(__file__).resolve().parent
    meta.to_parquet(BASE_DIR / f"data/{underlying}_{option_type}_surface_inputs.parquet", index=False)
    chain.to_parquet(BASE_DIR / f"data/{underlying}_{option_type}_surface_chain.parquet", index=False)

    print(f"[dataExtractor] Done. {len(chain)} contracts loaded.")