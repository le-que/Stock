import re
from io import StringIO
from datetime import datetime, timedelta
import requests
import pandas as pd
import requests 

#get the past historical prices of a stock
class YahooFinanceHistory:
    headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.0.3578.98 Safari/537.36'}
    quote_link = 'https://query1.finance.yahoo.com/v7/finance/download/{quote}?period1={dfrom}&period2={dto}&interval=1d&events=history'
    def __init__(self, symbol, years_back=7):
        self.dt = timedelta(days=365*years_back)
        self.symbol = symbol
    def get_quote(self):
        now = datetime.utcnow()
        dateto = int(now.timestamp())
        datefrom = int((now - self.dt).timestamp())
        url = self.quote_link.format(quote=self.symbol, dfrom=datefrom, dto=dateto)
        response = requests.get(url, headers=self.headers)
        # print(response.text)
        df = pd.DataFrame([sub.split(",") for sub in StringIO(response.text)])
        df.to_csv("stocks.csv", index=False, header=False)
        return pd.read_csv(StringIO(response.text), parse_dates=['Date'])
    
df = YahooFinanceHistory('AAPL', years_back=10).get_quote()
