import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

plt.style.use('seaborn')
data: pd.DataFrame = pd.read_csv('_')

#Finding Rolling Statistics
symbol = "AAPL"
data : pd.DataFrame = pd.DataFrame(data[symbol]).dropna()

window = 20

data['min'] = data[symbol].rolling(window=window).min()
data['max'] = data[symbol].rolling(window=window).max()
data['median'] = data[symbol].rolling(window=window).median()
data['mean'] = data[symbol].rolling(window=window).mean()
data['std'] = data[symbol].rolling(window=window).std()

data['ewma'] = data[symbol].ewm(halflife=0.5, min_periods=window).mean()

#Finding Correlation Statistics
symbol1, symbol2 = "AAPL", "MSFT"
data: pd.DataFrame = pd.DataFrame(data[symbol1, symbol2]).dropna()

rets : pd.DataFrame = np.log(data/data.shift(1)) #log returns
reg = np.polyfit(rets[symbol1], rets[symbol2], 1) #OLS regression
rets.corr() #correlation