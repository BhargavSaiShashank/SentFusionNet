import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import requests
import os
import datetime

# Updated ROOT_DIR for the Improved Branch
ROOT_DIR = r'C:\Users\shahs\FinalYear\improved_version'
os.makedirs(os.path.join(ROOT_DIR, 'data'), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'models'), exist_ok=True)
os.makedirs(os.path.join(ROOT_DIR, 'results'), exist_ok=True)

# MODERN PERIOD: 2020 to April 1st, 2026
start_date = datetime.datetime(2020, 1, 1)
end_date = datetime.datetime(2026, 4, 1)

# 1. Download S&P 500 Data (Modern)
print("Downloading modern S&P 500 data (2020-2026)...")
try:
    sp500 = yf.download('^GSPC', start=start_date, end=end_date)
    if isinstance(sp500.columns, pd.MultiIndex):
        sp500.columns = sp500.columns.get_level_values(0)
    sp500.to_csv(os.path.join(ROOT_DIR, 'data', 'sp500_raw.csv'))
    print(f"S&P 500 downloaded. Last close on {sp500.index[-1].date()}")
except Exception as e:
    print(f"S&P 500 failed: {e}")

# 2. Download Macroeconomic Indicators (Modern)
macro_series = {
    'Yield_10Y': 'DGS10',
    'EPUI_M': 'USEPUINDXM', 
    'EMUI': 'WLEMUINDXD',
    'BCI': 'BSCICP03USM665S',
    'CEI': 'UMCSENT',
    'ISM_PMI_M': 'INDPRO'
}

print("Downloading modern Macro data (2020-2026)...")
for name, sid in macro_series.items():
    try:
        df = web.DataReader(sid, 'fred', start_date, end_date)
        df.to_csv(os.path.join(ROOT_DIR, 'data', f'{name}.csv'))
        print(f"Downloaded {name} ({sid})")
    except Exception as e:
        print(f"Failed to download {sid}: {e}")

# 3. Sentiment Data Retrieval (Historical + Placeholder for Modern)
# Since the DJIA news set ended in 2016, we'll download it for general training 
# and use a new "Life Forecast" scraper for today's prediction.
sentiment_url = "https://raw.githubusercontent.com/gakudo-ai/open-datasets/refs/heads/main/Combined_News_DJIA.csv"
try:
    s_res = requests.get(sentiment_url)
    if s_res.status_code == 200:
        with open(os.path.join(ROOT_DIR, 'data', 'Combined_News_DJIA.csv'), 'wb') as f:
            f.write(s_res.content)
        print("Historical sentiment dataset downloaded for base training structure.")
except Exception as e:
    print(f"Error downloading sentiment: {e}")

print("Modern data retrieval complete.")
