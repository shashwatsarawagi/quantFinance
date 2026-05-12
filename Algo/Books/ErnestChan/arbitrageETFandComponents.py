import numpy as np
import pandas as pd

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant

from statsmodels.tsa.vector_ar.vecm import JohansenTestResult

from typing import List, Dict
from numpy.typing import NDArray


def johansen_cointegration_test(X: NDArray, det_order=0, k_ar_diff=1,significance_idx=1):
    result = coint_johansen(X, det_order, k_ar_diff)

    trace_stat = result.lr1[0]
    critical_value: float = result.cvt[0, significance_idx]

    is_cointegrated: bool = trace_stat > critical_value

    return result, is_cointegrated, result.evec[:, 0]


def estimate_half_life(spread: pd.Series) -> float:
    spread = pd.Series(spread).dropna()

    lagged = spread.shift(1).dropna()
    delta = spread.diff().dropna()

    lagged = lagged.loc[delta.index]

    model = OLS(delta.values, add_constant(lagged.values)).fit()

    beta = model.params[1]

    if beta >= 0:
        return np.inf

    half_life = -np.log(2) / beta

    return half_life


def etf_stock_arbitrage_strategy(
    daily_closes: pd.DataFrame,
    stocks: List[str],
    etf_closes: pd.Series,
    train_ratio:float =0.7,
    johansen_det_order : int =0,
    johansen_k_ar_diff : int =1,
) -> Dict:
    stock_df = pd.DataFrame(daily_closes, columns=stocks)
    etf_series = pd.Series(etf_closes, name="ETF")

    T = len(etf_series)

    split_idx = int(T * train_ratio)

    stock_train = stock_df.iloc[:split_idx]
    stock_test = stock_df.iloc[split_idx:]

    etf_train = etf_series.iloc[:split_idx]
    etf_test = etf_series.iloc[split_idx:]

    # Test each stock individually against ETF

    cointegrated_stocks: List[str] = []

    individual_results: Dict[str, Dict] = {}

    for symbol in stocks:

        pair_data = np.column_stack([stock_train[symbol], etf_train])

        try:
            result, is_coint, eigvec = johansen_cointegration_test(
                pair_data,
                det_order = johansen_det_order,
                k_ar_diff = johansen_k_ar_diff
            )

            individual_results[symbol] = {
                "cointegrated": is_coint,
                "eigenvector": eigvec,
                "trace_stat": result.lr1[0],
                "critical_value_95": result.cvt[0, 1],
            }

            if is_coint:
                cointegrated_stocks.append(symbol)

        except Exception as e:
            individual_results[symbol] = {
                "cointegrated": False,
                "error": str(e)
            }
    
    # Construct equal-weight long-only stock portfolio
    if len(cointegrated_stocks) == 0:
        raise ValueError("No individually cointegrated stocks found.")

    weights_stock_portfolio = np.ones(len(cointegrated_stocks))
    weights_stock_portfolio /= weights_stock_portfolio.sum()

    train_portfolio = (
        stock_train[cointegrated_stocks].values
        @ weights_stock_portfolio
    )

    test_portfolio = (
        stock_test[cointegrated_stocks].values
        @ weights_stock_portfolio
    )

    # Test portfolio cointegration with ETF
    portfolio_pair_train = np.column_stack([
        train_portfolio,
        etf_train
    ])

    _, portfolio_cointegrated, portfolio_eigvec = johansen_cointegration_test(
        portfolio_pair_train,
        det_order=johansen_det_order,
        k_ar_diff=johansen_k_ar_diff
    )

    if not portfolio_cointegrated:
        raise ValueError(
            "Equal-weight stock portfolio is NOT cointegrated with ETF."
        )

    # Build stationary spread

    # Johansen eigenvector gives:
    # w1 * Portfolio + w2 * ETF = stationary spread

    w_portfolio = portfolio_eigvec[0]
    w_etf = portfolio_eigvec[1]

    spread_train = (
        w_portfolio * train_portfolio
        + w_etf * etf_train.values
    )

    spread_test = (
        w_portfolio * test_portfolio
        + w_etf * etf_test.values
    )

    # Compute z-score signals
    spread_mean = np.mean(spread_train)
    spread_std = np.std(spread_train)

    zscore_test = (
        spread_test - spread_mean
    ) / spread_std

    signals = np.zeros(len(zscore_test))

    signals[zscore_test > 2] = -1
    signals[zscore_test < -2] = 1

    # Half-life estimation
    half_life = estimate_half_life(spread_train)


    return {
        "cointegrated_stocks": cointegrated_stocks,
        "individual_results": individual_results,

        "stock_portfolio_weights": dict(
            zip(cointegrated_stocks, weights_stock_portfolio)
        ),

        "portfolio_cointegrated": portfolio_cointegrated,

        "johansen_eigenvector": {
            "stock_portfolio_weight": w_portfolio,
            "etf_weight": w_etf
        },

        "spread_train": spread_train,
        "spread_test": spread_test,

        "zscore_test": zscore_test,
        "signals": signals,

        "half_life": half_life,

        "train_size": split_idx,
        "test_size": T - split_idx,
    }