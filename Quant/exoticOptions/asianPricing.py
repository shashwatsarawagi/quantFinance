import numpy as np

# Monte Carlo pricing of an arithmetic-average Asian call option.
def asian_arithmetic_call_mc(
    S0: float,
    K: float,
    r: float,
    sigma: float,
    T: float,
    steps: int,
    n_paths: int) -> float:
    """
    Monte Carlo pricing of an arithmetic-average Asian call option.
    """
    
    dt = T / steps
    
    # simulate Brownian increments
    Z = np.random.randn(n_paths, steps)
    
    # construct price paths
    S = np.zeros((n_paths, steps + 1))
    S[:, 0] = S0
    
    for t in range(1, steps + 1):
        S[:, t] = S[:, t-1] * np.exp(
            (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z[:, t-1]
        )
    
    # arithmetic average (excluding S0 is also common—adjust if needed)
    A = S[:, 1:].mean(axis=1)
    
    # payoff
    payoff = np.maximum(A - K, 0)
    
    # discount back
    price = np.exp(-r * T) * payoff.mean()
    
    return price