import os
import blpapi
from xbbg import blp
from dataclasses import dataclass
from datetime import date
from typing import List

import numpy as np
import pandas as pd
from pandas import DataFrame, Series

HIST_FIELDS = ["PX_LAST", "PX_OPEN", "PX_HIGH", "PX_LOW", "PX_VOLUME"]

GREEK_FIELDS = [
    "IVOL_MID",
    "DELTA_MID",
    "GAMMA_MID",
    "THETA_MID",
    "VEGA_MID",
    "RHO_MID",
]

FUNDAMENTAL_FIELDS = [
    "PX_TO_BOOK_RATIO",
    "PE_RATIO",
    "EQY_DVD_YLD_IND",
    "BETA_ADJUSTED",
    "CUR_MKT_CAP",
    "RETURN_COM_EQY",
    "EBITDA_TO_REVENUE",
]

@dataclass
class PortfolioPosition:
    ticker: str
    quantity: float
    cost_basis: float  = 0.0
    asset_class: str = "Equity"     # Equity | Option | Future | ETF
    option_type: str|None = None    # "Call" | "Put" | None
    strike: float|None = None
    expiry: date|None = None
    multiplier: float = 1.0

#felt easier to deal with the data this way than my impVolSurface approach. Recommended by AI!
@dataclass
class BloombergData: 
    tickers: List[str]
    prices: DataFrame
    returns: DataFrame   # log returns cause easier to work with
    greeks: DataFrame
    fundamentals: DataFrame
    last_prices: Series


# ── Main builder -----
def bloombergFetch(
    tickers: List[str],
    start: date,
    end: date | None = date.today(),
    ) -> BloombergData:

    prices = fetchHist(tickers, start, end)
    greeks = fetchFields(tickers, GREEK_FIELDS)
    fundamentals = fetchFields(tickers, FUNDAMENTAL_FIELDS)

    return dataBuilder(tickers, prices, greeks, fundamentals)

def fetchHist(tickers, start, end) -> DataFrame:
    blmData: DataFrame = blp.bdp(tickers=tickers, flds=HIST_FIELDS, start_date = start, end_date = end).to_pandas()
    blmData = blmData.reset_index().rename(columns={"index": "date"})
    blmData = blmData.pivot_table(index=["ticker", "date"], columns="field", values="value").reset_index()
    return blmData

def fetchFields(tickers, fields) -> pd.DataFrame:
    blmData: DataFrame = blp.bdp(tickers=tickers, flds=fields).to_pandas()
    blmData = blmData.reset_index().rename(columns={"index": "date"})
    blmData = blmData.pivot_table(index=["ticker", "date"], columns="field", values="value").reset_index()
    return blmData


def dataBuilder(tickers: List[str], prices: DataFrame, greeks: DataFrame, fundamentals: DataFrame) -> BloombergData:
    prices = prices.ffill().dropna(how="all")
    log_rets: DataFrame = DataFrame(np.log(prices / prices.shift(1))).dropna()
    last_prices = prices.iloc[-1]

    return BloombergData(
        tickers=tickers,
        prices=prices,
        returns=log_rets,
        greeks=greeks,
        fundamentals=fundamentals,
        last_prices=last_prices,
    )