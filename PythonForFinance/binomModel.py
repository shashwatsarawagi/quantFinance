import math
import numpy as np

def EuropeanPrice(asset: float, sigma: float, K: float, T: float, N: int, r: float, type: str = "Call") -> float:
    """Price a European call option using the binomial model."""
    dt = T / N
    discount = math.exp(-r * dt)

    temp1 = math.exp((r + sigma ** 2) * dt)
    temp2 = 0.5 * temp1 + 0.5 * discount

    u = temp2 + math.sqrt(temp2 ** 2 - 1)
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    S = np.zeros(N + 1)
    S[0] = asset
    for i in range(1, N + 1):
        for j in range(i, 0, -1): #j is the number of down moves
            S[j] = S[j - 1] * u
        S[0] = S[0] * d
    
    # Initialize option values at maturity
    asset_prices = np.zeros(N + 1)
    for j in range(N+1):
        asset_prices[j] = payoff(S[j], K, type)
    
    for n in range(N - 1, 0, -1):
        for j in range(n - 1):
            asset_prices[j] = discount * (p * asset_prices[j + 1] + (1 - p) * asset_prices[j])
    
    return asset_prices[0]

def AmericanPrice(asset: float, sigma: float, K: float, T: float, N: int, r: float, type: str = "Call") -> float:
    """Price an American call option using the binomial model."""
    dt = T / N
    discount = math.exp(-r * dt)

    temp1 = math.exp((r + sigma ** 2) * dt)
    temp2 = 0.5 * temp1 + 0.5 * discount

    u = temp2 + math.sqrt(temp2 ** 2 - 1)
    d = 1 / u
    p = (math.exp(r * dt) - d) / (u - d)

    # Initialize asset prices at maturity
    S = np.zeros((N + 1, N + 1))
    S[0, 0] = asset
    for i in range(1, N + 1):
        for j in range(i, 0, -1): #j is the number of down moves
            S[j, i] = S[j - 1, i - 1] * u
        S[0, i] = S[0, i - 1] * d
    
    # Initialize option values at maturity
    asset_prices = np.zeros((N + 1, N + 1))
    for j in range(N+1):
        asset_prices[j, N] = payoff(S[j, N], K, type)
    
    for n in range(N, 0, -1):
        for j in range(N - 1):
            asset_prices[j, n-1] = max(payoff(S[j, n-1], K, type), discount * (p * asset_prices[j + 1, n] + (1 - p) * asset_prices[j, n]))
    
    return asset_prices[0, 0]

def payoff(asset_price: float, K: float, type: str = "Call") -> float:
    if type == "Call":
        return max(asset_price - K, 0)
    else:
        return max(K - asset_price, 0)
