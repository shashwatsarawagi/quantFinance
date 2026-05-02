from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from scipy import stats

from pandas import DataFrame, Series


# ── Result container ──────────────────────────────────────────────────────────
@dataclass
class VaRResult:
    method: str
    confidence: float          # e.g. 0.95
    horizon_days: int
    var: float                 # positive number → potential loss
    es: float                  # Expected Shortfall / CVaR (always >= var)
    var_pct: float             # as % of portfolio value
    es_pct: float
    portfolio_value: float
    simulated_pnl: np.ndarray | None = None   # full distribution (MC)

    def __str__(self) -> str: # Coded completely by AI!
        return (
            f"[{self.method}] {int(self.confidence*100)}% VaR "
            f"({self.horizon_days}d): "
            f"£{self.var:,.0f} ({self.var_pct:.2%}) | "
            f"ES: £{self.es:,.0f} ({self.es_pct:.2%})"
        )


# ── Core engine ───────────────────────────────────────────────────────────────
class VaREngine:
    """
    Compute VaR and Expected Shortfall for a portfolio.
    """

    def __init__(self,
        returns: DataFrame,
        weights: Series | Dict,
        portfolio_value: float,
    ):
        self.returns = returns.copy()
        w: Series[Any] = Series(weights) if isinstance(weights, Dict) else weights.copy()
        
        # Align tickers
        common = self.returns.columns.intersection(w.index)
        self.returns = self.returns[common]
        w = w[common]
        self.weights = w / w.sum()
        self.portfolio_value = portfolio_value

        # Pre-compute portfolio daily log-returns
        self._port_rets: Series = self.returns @ self.weights

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _scale_to_horizon(daily_var: float, horizon: int) -> float:
        """Square-root-of-time scaling (conservative for non-normal)."""
        return daily_var * np.sqrt(horizon)

    @staticmethod
    def _es_from_tail(losses: np.ndarray, var: float) -> float:
        tail = losses[losses >= var]
        return float(tail.mean()) if len(tail) else var

    # ── 1. Historical Simulation ────────────
    def historical(self,
        confidence: float = 0.95,
        horizon_days: int = 1,
        lookback_days: int = 504,
    ) -> VaRResult:
        """
        Historical simulation using actual return distribution.
        No distributional assumption - captures fat tails and skew.
        """
        rets = self._port_rets.iloc[-lookback_days:]
        losses: Series[float] = -rets * self.portfolio_value  # positive = loss

        # Scale to horizon
        if horizon_days > 1:
            losses = losses * np.sqrt(horizon_days)

        var = float(np.percentile(losses, confidence * 100))
        es = self._es_from_tail(losses.to_numpy(), var)

        return VaRResult(
            method="Historical",
            confidence=confidence,
            horizon_days=horizon_days,
            var=max(var, 0.0),
            es=max(es, 0.0),
            var_pct=max(var, 0.0) / self.portfolio_value,
            es_pct=max(es, 0.0) / self.portfolio_value,
            portfolio_value=self.portfolio_value,
            simulated_pnl=(-losses).to_numpy(),
        )

    # ── 2. Parametric (Variance-Covariance) ─────────
    def parametric(self,
        confidence: float = 0.95,
        horizon_days: int = 1,
        lookback_days: int = 504,
    ) -> VaRResult:
        """
        Parametric VaR assuming multivariate normality.
        Fast, analytical, but underestimates tail risk in practice.
        """
        rets = self.returns.iloc[-lookback_days:]
        cov = rets.cov()
        mu = rets.mean()

        port_mu = float(self.weights @ mu)
        port_vol = float(np.sqrt(self.weights @ cov @ self.weights))

        z = stats.norm.ppf(confidence)
        daily_var_pct = -port_mu + z * port_vol
        daily_es_pct = -port_mu + port_vol * stats.norm.pdf(z) / (1 - confidence)

        var_pct = daily_var_pct * np.sqrt(horizon_days)
        es_pct = daily_es_pct * np.sqrt(horizon_days)

        var = var_pct * self.portfolio_value
        es = es_pct * self.portfolio_value

        return VaRResult(
            method="Parametric",
            confidence=confidence,
            horizon_days=horizon_days,
            var=max(var, 0.0),
            es=max(es, 0.0),
            var_pct=max(var_pct, 0.0),
            es_pct=max(es_pct, 0.0),
            portfolio_value=self.portfolio_value,
        )

    # ── 3. Monte Carlo ──────
    def monte_carlo(self,
        confidence: float = 0.95,
        horizon_days: int = 1,
        n_sims: int = 50_000,
        lookback_days: int = 504,
        seed: int = 42,
    ) -> VaRResult:
        """
        Monte Carlo simulation with correlated GBM.
        Best for non-linear payoffs (options, structured products).
        """
        rets = self.returns.iloc[-lookback_days:]
        mu = rets.mean().to_numpy()
        cov = rets.cov().to_numpy()
        n = len(self.weights)
        w = self.weights.to_numpy()

        rng = np.random.default_rng(seed)
        L = np.linalg.cholesky(cov + 1e-10 * np.eye(n))  # jitter for stability

        # Simulate horizon_days of returns, sum log-returns
        total_log_rets = np.zeros(n_sims)
        for _ in range(horizon_days):
            z = rng.standard_normal((n_sims, n))
            r = mu + z @ L.T
            total_log_rets += r @ w

        port_pnl = (np.exp(total_log_rets) - 1) * self.portfolio_value
        losses = -port_pnl

        var = float(np.percentile(losses, confidence * 100))
        es = self._es_from_tail(losses, var)

        return VaRResult(
            method="Monte Carlo",
            confidence=confidence,
            horizon_days=horizon_days,
            var=max(var, 0.0),
            es=max(es, 0.0),
            var_pct=max(var, 0.0) / self.portfolio_value,
            es_pct=max(es, 0.0) / self.portfolio_value,
            portfolio_value=self.portfolio_value,
            simulated_pnl=port_pnl,
        )

    # ── Stress Tests ──────────────────────────────────────────────────────
    def stress_test(self,
        scenarios: Dict[str, float] = {
            "COVID Crash (Mar 2020)": -0.34,
            "GFC Peak-to-Trough (2008-09)": -0.57,
            "Dot-com Bust (2000-02)": -0.49,
            "Black Monday (Oct 1987)": -0.23,
            "Rate Shock +200bps": -0.12,
            "Flash Crash (May 2010)": -0.09,
        },
    ) -> DataFrame:
        """
        Apply named shock scenarios (as % portfolio return) and compute loss.
        Default scenarios approximate historical crises.
        """
        
        rows = []
        for name, shock in scenarios.items():
            loss = -shock * self.portfolio_value
            rows.append( #Coded by AI!
                {
                    "Scenario": name,
                    "Portfolio Shock": f"{shock:.1%}",
                    "Estimated Loss (£)": f"£{loss:,.0f}",
                    "Estimated Loss (%)": f"{-shock:.1%}",
                }
            )
        return DataFrame(rows).set_index("Scenario")

    # ── Summary ───────────────────────────────────────────────────────────
    def full_report(self, #Coded by AI!
        confidence: float = 0.95,
        horizon_days: int = 1,
    ) -> DataFrame:
        hist = self.historical(confidence, horizon_days)
        param = self.parametric(confidence, horizon_days)
        mc = self.monte_carlo(confidence, horizon_days)

        rows = []
        for r in [hist, param, mc]:
            rows.append(
                {
                    "Method": r.method,
                    f"VaR (£)": f"£{r.var:,.0f}",
                    "VaR (%)": f"{r.var_pct:.2%}",
                    "ES/CVaR (£)": f"£{r.es:,.0f}",
                    "ES/CVaR (%)": f"{r.es_pct:.2%}",
                }
            )
        df = DataFrame(rows).set_index("Method")
        return df