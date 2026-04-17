import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import requests
import os
import datetime

# Create directories
ROOT_DIR = r'C:\Users\shahs\FinalYear\paper_replication'
os.makedirs(os.path.join(ROOT_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'results'), exist_ok=True)

start_date = datetime.datetime(2008, 1, 1)
end_date = datetime.datetime(2016, 7, 2)

# 1. Download S&P 500 Data
print("Downloading S&P 500 data...")
try:
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    sp500.to_csv(os.path.join(ROOT_DIR, 'data', 'sp500_raw.csv'))
    print("S&P 500 downloaded.")
except Exception as e:
    print(f"S&P 500 failed: {e}")

# 2. Download Macroeconomic Indicators from FRED
macro_series = {
    'Yield_10Y': 'DGS10',
    'EPUI_M': 'USEPUINDXM', 
    'EMUI': 'WLEMUINDXD',
    'BCI': 'BSCICP03USM665S',
    'CEI': 'UMCSENT', # University of Michigan Consumer Sentiment (Better available proxy)
    'ISM_PMI_M': 'INDPRO' # Industrial Production Index (Proxy for ISM)
}

print("Downloading Macro data from FRED...")
for name, sid in macro_series.items():
    try:
        df = web.DataReader(sid, 'fred', start_date, end_date)
        df.to_csv(os.path.join(ROOT_DIR, 'data', f'{name}.csv'))
        print(f"Downloaded {name} ({sid})")
    except Exception as e:
        print(f"Failed to download {sid}: {e}")

# 3. Sentiment Data
sentiment_url = "https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Combined_News_DJIA.csv"
print("Downloading Sentiment headlines...")
try:
    s_res = requests.get(sentiment_url)
    if s_res.status_code == 200:
        with open(os.path.join(ROOT_DIR, 'data', 'Combined_News_DJIA.csv'), 'wb') as f:
            f.write(s_res.content)
        print("Downloaded sentiment data.")
except Exception as e:
    print(f"Error downloading sentiment: {e}")

print("Data retrieval complete.")
