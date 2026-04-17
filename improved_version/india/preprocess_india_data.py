import pandas as pd
import numpy as np
import os
import ta

# Paths
ROOT_DIR = r'C:\Users\shahs\FinalYear\improved_version\india'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESSED_FILE = os.path.join(DATA_DIR, 'india_final_dataset.csv')

def main():
    print("--- PREPROCESSING INDIAN MARKET DATA ---")
    
    nifty = pd.read_csv(os.path.join(DATA_DIR, 'nifty50_raw.csv'), index_col=0, parse_dates=True)
    yield_in = pd.read_csv(os.path.join(DATA_DIR, 'india_yield.csv'), index_col=0, parse_dates=True)
    pmi_in = pd.read_csv(os.path.join(DATA_DIR, 'india_pmi.csv'), index_col=0, parse_dates=True)
    
    # Merge
    final_df = nifty.copy()
    if 'Adj Close' in final_df.columns:
        final_df['Price'] = final_df['Adj Close']
    else:
        final_df['Price'] = final_df['Close']
    final_df = final_df.join([yield_in, pmi_in], how='left')
    
    final_df.ffill(inplace=True); final_df.bfill(inplace=True)
    
    # Technical Indicators
    print("Computing Technical Signals for Nifty 50...")
    final_df['Log_Return'] = np.log(final_df['Price'] / final_df['Price'].shift(1))
    final_df['EMA'] = ta.trend.ema_indicator(final_df['Price'], window=20)
    final_df['MACD_Line'] = ta.trend.macd(final_df['Price'])
    final_df['RSI'] = ta.momentum.rsi(final_df['Price'], window=14)
    
    # Indian Sentiment Proxy (Simulated for this branch extension)
    final_df['Sentiment_Score'] = np.random.uniform(-0.1, 0.1, size=len(final_df))
    
    # Lagging Features (Targeting Stationary Returns)
    feature_cols = ['Log_Return', 'EMA', 'MACD_Line', 'RSI', 'India_10Y_Yield', 'India_PMI', 'Sentiment_Score']
    for c in feature_cols:
        final_df[f'{c}_lag1'] = final_df[c].shift(1)
        
    final_df['Target_Return'] = final_df['Log_Return']
    final_df.dropna(inplace=True)
    
    final_df.to_csv(PROCESSED_FILE)
    print(f"Indian Preprocessing Complete: {len(final_df)} rows ready for Nifty 50 training.")

if __name__ == "__main__":
    main()
