import os
import blpapi
from xbbg import blp
import pandas as pd

DATA_DIR = "./data"

tickers = ["NVDA US Equity", "AAPL US Equity"]
fields = ["High", "Low", "Last_Price"]
start_date = '2023-09-01'
end_date = '2023-09-20'

hist_tick_data = blp.bdh(tickers=tickers, flds=fields, start_date=start_date, end_date=end_date)
filename = f'tickdata_{start_date}_{end_date}.csv'
hist_tick_data.write_csv(f'{DATA_DIR}/{filename}')