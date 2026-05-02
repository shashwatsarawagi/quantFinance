from math import log, sqrt, exp
from scipy import stats
from typing import Dict

def bsm_callGreeks(S: float, K: float, T: float, r: float, sigma: float) -> Dict[str, float]:
    greeks: dict[str, float] = {}
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    greeks['delta'] = stats.norm.cdf(d1, 0.0, 1.0).item()
    greeks['gamma'] = (stats.norm.pdf(d1).item())/(sigma*S*sqrt(T))
    greeks['theta'] = (-(sigma*S*stats.norm.pdf(d1).item())/(2*sqrt(T)))
    greeks['speed'] = (-stats.norm.pdf(d1).item())/(sigma**2 * S**2 * T)
    greeks['vega'] = S * stats.norm.pdf(d1, 0.0, 1.0).item() * sqrt(T)
    greeks['rho'] = K*T*exp(-r*T)*stats.norm.cdf(d2).item()

    return greeks


def bsm_callPrice(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    C: float = (S * stats.norm.cdf(d1, 0.0, 1.0).item() - 
         K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0).item())

    return C

def bsm_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega : float = S * stats.norm.pdf(d1, 0.0, 1.0).item() * sqrt(T)
    return vega


def bsm_callImpVol(S: float, K: float, T: float, r: float, C: float, sigma_est: float = 0.3, it : int = 100) -> float:
    for _ in range(it):
        sigma_est -= ((bsm_callPrice(S, K, T, r, sigma_est) - C) / bsm_vega(S, K, T, r, sigma_est))
    return sigma_est