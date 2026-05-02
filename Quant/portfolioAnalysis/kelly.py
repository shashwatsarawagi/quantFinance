"""
Kelly fractions > 1 imply leverage. Always inspect the fractional output.
"""
from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize


# ── Result containers ─────────────────────────────────────────────────────────
@dataclass
class KellyResult:
    method: str
    ticker: str | None
    full_kelly: float       # full Kelly fraction (can exceed 1 -> leverage)
    half_kelly: float
    quarter_kelly: float
    recommended_fraction: float   # typically half-Kelly for safety
    implied_edge: float           # expected log-growth at full Kelly
    note: str = ""

    def __str__(self) -> str: #Coded by AI!
        return (
            f"[{self.method}] {self.ticker or 'Portfolio'}\n"
            f"  Full Kelly      : {self.full_kelly:.4f}  ({self.full_kelly:.1%})\n"
            f"  Half Kelly      : {self.half_kelly:.4f}  ({self.half_kelly:.1%})\n"
            f"  Recommended (½K): {self.recommended_fraction:.4f}  ({self.recommended_fraction:.1%})\n"
            f"  Implied Edge    : {self.implied_edge:.4f}  ({self.implied_edge:.2%} log-growth/trade)\n"
            + (f"  Note: {self.note}" if self.note else "")
        )


@dataclass
class MultiAssetKelly:
    tickers: List[str]
    weights: pd.Series       # full-Kelly optimal weights
    half_kelly_weights: pd.Series
    expected_log_growth: float
    leverage: float          # sum of absolute weights

    def __str__(self) -> str: #Coded by AI!
        lines = [
            "── Multi-Asset Kelly Optimal Allocation ─────────────",
            f"  Expected log-growth (full K): {self.expected_log_growth:.4f}",
            f"  Implied leverage            : {self.leverage:.2f}x",
            "",
            "  Full Kelly Weights:",
        ]
        for t, w in self.weights.items():
            lines.append(f"    {t:<30} {w:+.4f}  ({w:+.1%})")
        lines.append("\n  Half Kelly Weights:")
        for t, w in self.half_kelly_weights.items():
            lines.append(f"    {t:<30} {w:+.4f}  ({w:+.1%})")
        lines.append("─────────────────────────────────────────────────────")
        return "\n".join(lines)


# ── Kelly engine ─────────
class KellyCriterion:
    """
    Kelly Criterion position sizing.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        rf_annual: float = 0.0525,
    ):
        self.returns = returns.copy()
        self.rf_annual = rf_annual
        self.rf_daily = (1 + rf_annual) ** (1 / 252) - 1

    # ── 1. Discrete Kelly (win/loss model) ─────────
    def discrete(
        self,
        ticker: str,
        win_prob: float | None = None,
        win_return: float | None = None,
        loss_return: float | None = None,
        kelly_fraction: float = 0.5,
    ) -> KellyResult:
        """
        Classic Kelly formula: f* = (p*b - q) / b
        where b = win_return / |loss_return|, p = P(win), q = 1 - p.

        If win_prob / win_return / loss_return are not supplied,
        they are estimated from the return series using the up/down split.
        """
        rets = self.returns[ticker].dropna()

        if win_prob is None or win_return is None or loss_return is None:
            wins = rets[rets > 0]
            losses = rets[rets <= 0]
            p = len(wins) / len(rets)
            b_win = float(wins.mean()) if len(wins) else 0.01
            b_loss = abs(float(losses.mean())) if len(losses) else 0.01
        else:
            p = win_prob
            b_win = win_return
            b_loss = abs(loss_return)

        q = 1 - p
        b = b_win / b_loss  # odds ratio
        full_k = (p * b - q) / b if b > 0 else 0.0
        full_k = max(full_k, 0.0)  # no short signal from this formula

        edge = p * np.log(1 + full_k * b_win) + q * np.log(1 - full_k * b_loss)

        note = ""
        if full_k > 1:
            note = "Full Kelly > 1 implies leverage. Use fractional Kelly."
        if full_k == 0:
            note = "No edge detected. Kelly fraction is zero."

        return KellyResult(
            method="Discrete",
            ticker=ticker,
            full_kelly=full_k,
            half_kelly=full_k / 2,
            quarter_kelly=full_k / 4,
            recommended_fraction=full_k * kelly_fraction,
            implied_edge=edge,
            note=note,
        )

    # ── 2. Continuous Kelly (mean-variance) ──────────
    def continuous(self,
        ticker: str,
        lookback: int = 252,
        kelly_fraction: float = 0.5,
    ) -> KellyResult:
        """
        Continuous Kelly for log-normal asset: f* = mu / sigma²
        where mu = excess mean daily return, sigma = daily volatility.
        Annualises both before dividing.
        """
        rets = self.returns[ticker].iloc[-lookback:].dropna()
        mu = float(rets.mean() * 252) - self.rf_annual  # annualised excess return
        sigma2 = float(rets.std() ** 2 * 252)           # annualised variance

        full_k = mu / sigma2 if sigma2 > 0 else 0.0
        edge = mu * full_k - 0.5 * sigma2 * full_k**2   # log-growth at full K

        note = ""
        if full_k > 2:
            note = "Very high Kelly fraction — excess return likely overfitted. Use half-Kelly."
        if full_k < 0:
            note = "Negative Kelly: asset has negative risk-adjusted edge. Avoid or short."

        return KellyResult(
            method="Continuous",
            ticker=ticker,
            full_kelly=full_k,
            half_kelly=full_k / 2,
            quarter_kelly=full_k / 4,
            recommended_fraction=full_k * kelly_fraction,
            implied_edge=edge,
            note=note,
        )

    # ── 3. Multi-asset Kelly (covariance-based) ───────────────────────────
    def multi_asset(self,
        tickers: List[str] | None = None,
        lookback: int = 252,
        allow_short: bool = False,
        max_leverage: float = 1.5,
        kelly_fraction: float = 0.5,
    ) -> MultiAssetKelly:
        """
        Multi-asset Kelly via quadratic program:
            max  w^T μ - ½ w^T Σ w   (log-growth approximation)
        subject to:
            Σ|w_i| ≤ max_leverage
            w_i ≥ 0 if not allow_short
        """
        tickers = tickers or list(self.returns.columns)
        rets = self.returns[tickers].iloc[-lookback:].dropna()
        mu = rets.mean().array[0] * 252 - self.rf_annual
        cov = rets.cov().array[0] * 252
        n = len(tickers)

        def neg_log_growth(w):
            return -(w @ mu - 0.5 * w @ cov @ w)

        def neg_lg_grad(w):
            return -(mu - cov @ w)

        bounds = (
            [(-max_leverage, max_leverage)] * n
            if allow_short
            else [(0.0, max_leverage)] * n
        )
        constraints = [
            {"type": "ineq", "fun": lambda w: max_leverage - np.sum(np.abs(w))},
        ]

        w0 = np.ones(n) / n
        res = minimize(
            neg_log_growth,
            w0,
            jac=neg_lg_grad,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-9, "maxiter": 1000},
        )

        full_w = pd.Series(res.x, index=tickers)
        half_w = full_w * kelly_fraction
        log_growth = float(-res.fun)
        leverage = float(np.abs(full_w).sum())

        return MultiAssetKelly(
            tickers=tickers,
            weights=full_w,
            half_kelly_weights=half_w,
            expected_log_growth=log_growth,
            leverage=leverage,
        )

    # ── Summary table ───────
    def summary_table(self,
        tickers: List[str] | None = None,
        lookback: int = 252,
    ) -> pd.DataFrame:
        """Return a DataFrame of continuous Kelly results for all tickers."""
        tickers = tickers or list(self.returns.columns)
        rows = []
        for t in tickers: #Coded by AI!
            r = self.continuous(t, lookback)
            rows.append(
                {
                    "Ticker": t,
                    "Ann. Return": f"{(self.returns[t].iloc[-lookback:].mean()*252):.2%}",
                    "Ann. Vol": f"{(self.returns[t].iloc[-lookback:].std()*np.sqrt(252)):.2%}",
                    "Full Kelly": f"{r.full_kelly:.3f}",
                    "Half Kelly": f"{r.half_kelly:.3f}",
                    "Implied Edge": f"{r.implied_edge:.4f}",
                    "Note": r.note,
                }
            )
        return pd.DataFrame(rows).set_index("Ticker")
