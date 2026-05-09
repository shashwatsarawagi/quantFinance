from typing import List, Tuple
from scipy import stats
from scipy.stats._stats_py import LinregressResult
import numpy as np
import numpy.typing as npt

#We introduce cointegration into our strategy
def ma(x: npt.ArrayLike, lookback) -> npt.NDArray:
    x = np.asarray(x, dtype=float)
    cumsum = np.cumsum(x)
    cumsum = np.insert(cumsum, 0, 0)
    return (cumsum[lookback:] - cumsum[:-lookback]) / lookback


def mstd(x, lookback):
    x = np.asarray(x, dtype=float)

    ma1 = ma(x, lookback)
    ma2 = ma(x**2, lookback)

    var = ma2 - ma1**2
    var = np.maximum(var, 0)  # numerical safety

    return np.sqrt(var)

def lag(x, roll):
    x_lag = np.roll(x, roll, axis=0)

    # fix wrap-around
    x_lag[0:roll] = np.full(x_lag[0:roll].shape, np.nan)

    return x_lag

def zScore(x: npt.NDArray[np.float64], lookback: int = 0) -> npt.NDArray[np.float64]:
    return (x - ma(x, lookback))/mstd(x, lookback)

def forward_fill(arr):
    arr = np.asarray(arr, dtype=float)

    idx = np.where(~np.isnan(arr), np.arange(len(arr)), 0)
    idx = np.maximum.accumulate(idx)

    return arr[idx]





#First, we implement a simple dynamicMeanReverson
def dynamicMeanReversion(xprices: npt.ArrayLike, yprices: npt.ArrayLike, lookback: int) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, float]:
    xprices = np.array(xprices)
    yprices = np.array(yprices)
    hedgeRatio: npt.NDArray = np.zeros_like(xprices, dtype=float)

    for t in range(lookback, len(xprices)):
       slope, _, _, _, _ = stats.linregress(xprices[t-lookback+1 : t+1], yprices[t-lookback+1 : t])
       hedgeRatio[t] = slope

    port = yprices - hedgeRatio * xprices

    y2 = np.column_stack((xprices, yprices))

    numUnits = -zScore(port, lookback)
    
    hedge = np.column_stack((hedgeRatio, -np.ones_like(hedgeRatio)))
    positions = numUnits * hedge * y2

    returns = (y2 - lag(y2, 1)) / lag(y2, 1)
    pnl = np.nansum(lag(positions, 1) * returns, axis=1)

    return hedgeRatio, port, numUnits, pnl

def boilingerBands(xprices: npt.ArrayLike, yprices: npt.ArrayLike, lookback: int,
                   entryZ: int = 1, exitZ: int = 1)  -> Tuple[npt.NDArray, npt.NDArray, float]:
    
    hedgeRatio, port, numUnits, _ = dynamicMeanReversion(xprices, yprices, lookback)
    zScor = zScore(port, lookback)

    longsEntry = zScor < (-entryZ)
    longsExit = zScor >= (-exitZ)

    shortsEntry = zScor > entryZ
    shortsExit = zScor <= exitZ

    numUnitsLong = np.full(port.shape, np.nan)
    numUnitsLong[0] = 0
    numUnitsLong[longsEntry] = 1
    numUnitsLong[longsExit] = 0
    numUnitsLong = forward_fill(numUnitsLong)


    numUnitsShort = np.full(port.shape, np.nan)
    numUnitsShort[0] = 0
    numUnitsShort[shortsEntry] = 1
    numUnitsShort[shortsExit] = 0
    numUnitsShort = forward_fill(numUnitsShort)

    numUnits = numUnitsLong + numUnitsShort


    y2 = np.column_stack((xprices, yprices))
    hedge = np.column_stack((hedgeRatio, -np.ones_like(hedgeRatio)))
    positions = numUnits * hedge * y2

    returns = (y2 - lag(y2, 1)) / lag(y2, 1)
    pnl = np.nansum(lag(positions, 1) * returns, axis=1)

    return port, numUnits, pnl


import numpy as np



def kalman_filter(xprices, yprices, delta=1e-4):
    x = np.asarray(xprices, dtype=float)
    y = np.asarray(yprices, dtype=float)
    ypred = np.zeros_like(yprices)

    n = len(x)

    beta = np.zeros_like(x)
    p = np.zeros_like(np.array([[0, 0], [0, 0]]))

    e = np.zeros_like(y)
    q = np.zeros_like(y)

    beta[1] = 0.0
    p[1] = 1.0

    # fixed state variance (your definition)
    Vw = (delta / (1 - delta)) * np.diag(np.ones(2))
    Ve = 1e-3

    for t in range(1, n):
        if t>1:
            beta[t] = beta[t-1]
            r = p + Vw
        else:
            r = p
        
        ypred[t] = np.dot(x[t], beta[t])
        q[t] = np.dot(np.dot(r, x[t]), x[t])

        e[t] = y[t] - ypred[t]

        k = np.dot(r, x[t]) * (1/q[t])
        beta[t] = beta[t] + k*e[t]
        p = r - np.dot(k, x[t])*r

    y2 = np.column_stack((x[1], y))
    longsEntry = e < -np.sqrt(q)
    longsExit = e > -np.sqrt(q)

    shortsEntry = e > np.sqrt(q)
    shortsExit = e < np.sqrt(q)

    hedgeRatio = beta

    numUnitsLong = np.full(p.shape, np.nan)
    numUnitsLong[0] = 0
    numUnitsLong[longsEntry] = 1
    numUnitsLong[longsExit] = 0
    numUnitsLong = forward_fill(numUnitsLong)


    numUnitsShort = np.full(p.shape, np.nan)
    numUnitsShort[0] = 0
    numUnitsShort[shortsEntry] = 1
    numUnitsShort[shortsExit] = 0
    numUnitsShort = forward_fill(numUnitsShort)

    numUnits = numUnitsLong + numUnitsShort


    y2 = np.column_stack((xprices, yprices))
    hedge = np.column_stack((hedgeRatio, -np.ones_like(hedgeRatio)))
    positions = numUnits * hedge * y2

    returns = (y2 - lag(y2, 1)) / lag(y2, 1)
    pnl = np.nansum(lag(positions, 1) * returns, axis=1)

    return p, numUnits, pnl

#Pg 81