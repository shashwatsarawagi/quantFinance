from math import log, sqrt, exp
from scipy import stats

def bsm_callPrice(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    C: float = (S * stats.norm.cdf(d1, 0.0, 1.0).item() - 
         K * exp(-r * T) * stats.norm.cdf(d2, 0.0, 1.0).item())

    return C

def bsm_vega(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    vega : float = S * stats.norm.cdf(d1, 0.0, 1.0).item() * sqrt(T)
    return vega


def bsm_callImpVol(S: float, K: float, T: float, r: float, C: float, sigma_est: float = 0.3, it : int = 100) -> float:
    for _ in range(it):
        sigma_est -= ((bsm_callPrice(S, K, T, r, sigma_est) - C) / bsm_vega(S, K, T, r, sigma_est))
    return sigma_est