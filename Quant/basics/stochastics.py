
import numpy.random as npr
import math
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')


def create_plot(x, y, styles, labels, axlabels) -> None:
    plt.figure(figsize=(10, 6))
    for i in range(len(x)):
        plt.plot(x[i], y[i], styles[i], label=labels[i])
        plt.xlabel(axlabels[0])
        plt.ylabel(axlabels[1])
    plt.legend(loc=0)


"""Approximation"""


def f(x: np.ndarray) -> np.ndarray:
    return np.sin(x) + 0.5 * x


x = np.linspace(-2 * np.pi, 2 * np.pi, 50)
# create_plot([x], [f(x)], ['b'], ['f(x)'], ['x', 'f(x)'])


degree = 7
reg = np.polyfit(x, f(x), degree)
ry = np.polyval(reg, x)
# create_plot([x, x], [f(x), ry], ['b', 'r.'], ['f(x)', f"Degree {degree}"], ['x', 'f(x)'])


"""Stochastics"""
# Random Numbers

sampleSize = 500
stdNorm = npr.standard_normal(sampleSize)
normal = npr.normal(100, 20, sampleSize)
chiSquare = npr.chisquare(0.5, sampleSize)
poisson = npr.poisson(1, sampleSize)

# BSM Testing
S0, r, sigma, T = 100, 0.05, 0.25, 2.0
I = 1000

ST1 = S0 * np.exp((r - 0.5 * sigma**2) * T +
                  sigma * math.sqrt(T) * npr.standard_normal(I))


# Euler discretisation of Sqr Root diffusion
x0, kappa, theta, sigma = 0.05, 3.0, 0.02, 0.1
I, M = 10000, 50
dt = T / M


def srd_euler():
    xh = np.zeros((M + 1, I))
    x = np.zeros_like(xh)
    xh[0], x[0] = x0, x0

    for t in range(1, M + 1):
        xh[t] = (xh[t - 1] + kappa * (theta - np.maximum(xh[t - 1],
                                                         0)) * dt + sigma * np.sqrt(np.maximum(xh[t - 1],
                                                                                               0)) * math.sqrt(dt) * npr.standard_normal(I))
    x = np.maximum(xh, 0)
    return x


# Jump diffusions
S0, r, sigma, lamb, mu, delta = 100, 0.05, 0.2, 0.75, -0.6, 0.25
rj = lamb * (math.exp(mu + 0.5 * delta**2) - 1)

T, M, I = 1, 50, 10000
dt = T / M

S = np.zeros((M + 1, I))
S[0] = S0
sn1 = npr.standard_normal((M + 1, I))
sn2 = npr.standard_normal((M + 1, I))
poi = npr.poisson(lamb * dt, (M + 1, I))

for t in range(1, M + 1):
    S[t] = S[t - 1] * (np.exp((r - rj - 0.5 * sigma**2) * dt +
                              sigma * math.sqrt(dt) * sn1[t]) +
                       (np.exp(mu + delta * sn2[t]) - 1) * poi[t])
    S[t] = np.maximum(S[t], 0)

plt.figure(figsize=(10, 6))
plt.plot(S[:, :10], lw=1.5)


# European Options implementation
def gen_sn(M, I, anti_paths = True, mo_match = True):
    if anti_paths is True:
        sn = npr.standard_normal((M + 1, int(I / 2)))
        sn = np.concatenate((sn, -sn), axis=1)
    else:
        sn = npr.standard_normal((M + 1, I))
    
    if mo_match is True:
        sn = (sn - sn.mean()) / sn.std()
    return sn
def gbm_mcs_dyna(K, option='call'):  # dynamic geometric Brownian motion MonteCarlo simulator
    dt = T / M
    S = np.zeros((M + 1, I))
    S[0] = S0
    sn = gen_sn(M, I)
    
    for t in range(1, M+1):
        S[t] = S[t-1] * np.exp((r - 0.5 * sigma ** 2) * dt
                               + sigma * math.sqrt(dt) * sn[t])
    
    if option == "call":
        hT = np.maximum(S[-1] - K, 0)
    else:
        hT = np.maximum(K - S[-1], 0)
    
    C0 = math.exp(-r * T) * np.mean(hT)
    return C0