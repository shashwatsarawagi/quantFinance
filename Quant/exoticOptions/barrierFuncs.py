from math import log, sqrt, exp
import math
from scipy import stats
from typing import Dict, List

# Remember. If it is a call option then we are okay with S > K. So if K < Sd for a down, 
# then its okay cause we still make money when K < Sd < S. But if K < Su for an up, 
# then we are not okay cause an up and out needs S > K > Su to make money which can't 
# happen for an up barrier. Hence, the dynamics are complex for this situation.

def norm(x: float) -> float:
    return stats.norm.cdf(x, 0.0, 1.0).item()
def barrier_callprice(S: float, K: float, T: float, r: float, sigma: float, Sb: float, type: List[str] = ["down", "out"]) -> float:
    a = (Sb/S)**(2*r/sigma**2 - 1)
    b = (Sb/S)**(2*r/sigma**2 + 1)
    
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = (log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d3 = (log(S / Sb) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d4 = (log(S / Sb) + (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d5 = (log(S / Sb) - (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d6 = (log(S / Sb) - (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d7 = (log(S*K / Sb**2) - (r - 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d8 = (log(S*K / Sb**2) - (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))

    E = K*math.exp(-r*T)

    C = 0.0

    if type[0] == "down" and type[1] == "out":
        if K > Sb:
            C = S * (norm(d1) - b*(1-norm(d8))) - E*(norm(d2) - a*(1-norm(d7)))
        else:
            C = S * (norm(d3) - b*(1-norm(d6))) - E*(norm(d4) - a*(1-norm(d5)))
    
    elif type[0] == "down" and type[1] == "in":
        if K > Sb:
            C = S * b * (1-norm(d8)) - E * a * (1-norm(d7))
        else:
            C = S * (norm(d1) - norm(d3) + b*(1-norm(d6))) - E * (norm(d2) - norm(d4) + a*(1-norm(d5)))
        
    elif type[0] == "up" and type[1] == "out":
        C = S*(norm(d1) - norm(d3) - b*(norm(d6) - norm(d8))) - E*(norm(d2) - norm(d4) - a*(norm(d5) - norm(d7)))
    
    elif type[0] == "up" and type[1] == "in":
        C = S * (norm(d3)+b*(norm(d6) - norm(d8))) - E * (norm(d4) + a*(norm(d5) - norm(d7)))
        
    return C