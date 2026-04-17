import pandas as pd
import numpy as np
import os
import ta
import torch
from transformers import pipeline
from tqdm import tqdm

# Files and paths
DATA_DIR = r'C:\Users\shahs\FinalYear\paper_replication\data'
PROCESSED_FILE = os.path.join(DATA_DIR, 'final_dataset.csv')

def safe_read_fred(path):
    for date_col in ['DATE', 'observation_date', 'Date']:
        try:
            df = pd.read_csv(path, index_col=date_col, parse_dates=True)
            return df
        except (ValueError, KeyError):
            continue
    df = pd.read_csv(path)
    if 'DATE' in df.columns.to_list():
        df.set_index('DATE', inplace=True)
    elif 'observation_date' in df.columns.to_list():
        df.set_index('observation_date', inplace=True)
    df.index = pd.to_datetime(df.index)
    return df

def load_data():
    print("Loading datasets...")
    sp500 = pd.read_csv(os.path.join(DATA_DIR, 'sp500_raw.csv'), index_col='Date', parse_dates=True)
    yield_10y = safe_read_fred(os.path.join(DATA_DIR, 'Yield_10Y.csv'))
    emui = safe_read_fred(os.path.join(DATA_DIR, 'EMUI.csv'))
    bci = safe_read_fred(os.path.join(DATA_DIR, 'BCI.csv'))
    cei = safe_read_fred(os.path.join(DATA_DIR, 'CEI.csv'))
    epui_m = safe_read_fred(os.path.join(DATA_DIR, 'EPUI_M.csv'))
    pmi = safe_read_fred(os.path.join(DATA_DIR, 'ISM_PMI_M.csv'))
    sentiment_df = pd.read_csv(os.path.join(DATA_DIR, 'Combined_News_DJIA.csv'))
    sentiment_df['Date'] = pd.to_datetime(sentiment_df['Date'], dayfirst=True) 
    sentiment_df.set_index('Date', inplace=True)
    return sp500, yield_10y, emui, bci, cei, epui_m, pmi, sentiment_df

def preprocess_sentiment(df):
    print("Processing high-precision sentiment (DistilBERT - Match Scenario A)...")
    device = 0 if torch.cuda.is_available() else -1
    # Paper uses DistilBERT-base-uncased
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
    
    cols = [f'Top{i}' for i in range(1, 26)]
    for col in cols:
        df[col] = df[col].fillna('').astype(str).str.replace(r"^b['\"]", '', regex=True).str.replace(r"['\"]$", '', regex=True)
    
    df['Combined_News'] = df[cols].agg(' '.join, axis=1)
    
    print("Generating transformer scores (2-5 minutes)...")
    texts = df['Combined_News'].tolist()
    scores = []
    batch_size = 16
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch = [t[:512] for t in texts[i : i + batch_size]]
        results = sentiment_pipeline(batch)
        for r in results:
            s = r['score']
            if r['label'] == 'NEGATIVE':
                s = -s
            scores.append(s)
            
    df['Sentiment_Score'] = scores
    return df[['Sentiment_Score']]

def main():
    sp500, yield_10y, emui, bci, cei, epui_m, pmi, s_df = load_data()
    s_scores = preprocess_sentiment(s_df)
    
    for df in [yield_10y, emui, bci, cei, epui_m, pmi]:
        df.replace('.', np.nan, inplace=True)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    final_df = sp500.copy()
    final_df = final_df.join(yield_10y, how='left')
    final_df = final_df.join(emui, how='left')
    final_df = final_df.join(bci, how='left', rsuffix='_bci')
    final_df = final_df.join(cei, how='left', rsuffix='_cei')
    final_df = final_df.join(epui_m, how='left', rsuffix='_epui')
    final_df = final_df.join(pmi, how='left', rsuffix='_pmi')
    final_df = final_df.join(s_scores, how='left')
    
    final_df.ffill(inplace=True)
    final_df.bfill(inplace=True)
    
    rename_cols = {
        'Adj Close': 'Adj_Close', 'DGS10': 'Yield_10Y', 'WLEMUINDXD': 'EMUI',
        'BSCICP03USM665S': 'BCI', 'UMCSENT': 'CEI', 'USEPUINDXM': 'EPUI', 'INDPRO': 'PMI'
    }
    final_df.rename(columns=rename_cols, inplace=True)
    if 'Adj_Close' not in final_df.columns:
        final_df['Adj_Close'] = final_df['Close']

    print("Calculated technical indicators...")
    final_df['EMA'] = ta.trend.ema_indicator(final_df['Adj_Close'], window=20)
    final_df['MACD_Line'] = ta.trend.macd(final_df['Adj_Close'])
    final_df['MACD_Signal'] = ta.trend.macd_signal(final_df['Adj_Close'])
    final_df['MACD_Diff'] = ta.trend.macd_diff(final_df['Adj_Close'])
    final_df['RSI'] = ta.momentum.rsi(final_df['Adj_Close'], window=14)
    final_df['Stoch_K'] = ta.momentum.stoch(final_df['High'], final_df['Low'], final_df['Adj_Close'], window=14, smooth_window=3)
    final_df['Stoch_D'] = ta.momentum.stoch_signal(final_df['High'], final_df['Low'], final_df['Adj_Close'], window=14, smooth_window=3)
    final_df['Williams_R'] = ta.momentum.williams_r(final_df['High'], final_df['Low'], final_df['Adj_Close'], lbp=14)

    cols_to_lag = ['Adj_Close', 'EMA', 'MACD_Line', 'MACD_Signal', 'MACD_Diff', 'RSI', 
                   'Stoch_K', 'Stoch_D', 'Williams_R', 'Yield_10Y', 'EMUI', 'BCI', 
                   'CEI', 'EPUI', 'PMI', 'Sentiment_Score']
    
    for c in cols_to_lag:
        if c in final_df.columns:
            final_df[f'{c}_lag1'] = final_df[c].shift(1)
            
    final_df['Target_Price'] = final_df['Adj_Close']
    final_df.dropna(inplace=True)
    final_df = final_df.round(6)
    
    final_df.to_csv(PROCESSED_FILE)
    print(f"Replication Complete (DistilBERT): {PROCESSED_FILE}")

if __name__ == "__main__":
    main()
