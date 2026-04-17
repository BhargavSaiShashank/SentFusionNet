import pandas as pd
import numpy as np
import os
import ta
import torch
import warnings
warnings.filterwarnings('ignore')

# Files and paths (Improved Version branch)
ROOT_DIR = r'C:\Users\shahs\FinalYear\improved_version'
DATA_DIR = os.path.join(ROOT_DIR, 'data')
PROCESSED_FILE = os.path.join(DATA_DIR, 'final_dataset.csv')

def load_data():
    print("Loading modern datasets (2020-2026)...")
    sp500 = pd.read_csv(os.path.join(DATA_DIR, 'sp500_raw.csv'), index_col='Date', parse_dates=True)
    yield_10y = pd.read_csv(os.path.join(DATA_DIR, 'Yield_10Y.csv'), index_col=0, parse_dates=True)
    emui = pd.read_csv(os.path.join(DATA_DIR, 'EMUI.csv'), index_col=0, parse_dates=True)
    bci = pd.read_csv(os.path.join(DATA_DIR, 'BCI.csv'), index_col=0, parse_dates=True)
    cei = pd.read_csv(os.path.join(DATA_DIR, 'CEI.csv'), index_col=0, parse_dates=True)
    epui_m = pd.read_csv(os.path.join(DATA_DIR, 'EPUI_M.csv'), index_col=0, parse_dates=True)
    pmi = pd.read_csv(os.path.join(DATA_DIR, 'ISM_PMI_M.csv'), index_col=0, parse_dates=True)
    
    s_df = pd.DataFrame(index=sp500.index)
    s_df['Sentiment_Score'] = 0.0

    return sp500, yield_10y, emui, bci, cei, epui_m, pmi, s_df

def main():
    sp500, yield_10y, emui, bci, cei, epui_m, pmi, s_df = load_data()
    
    # --- RENAME PROTECTION ---
    # Merge and standardize columns
    final_df = sp500.copy()
    if 'Adj Close' in final_df.columns:
        final_df['Price'] = final_df['Adj Close']
    elif 'Adj_Close' in final_df.columns:
        final_df['Price'] = final_df['Adj_Close']
    else:
        final_df['Price'] = final_df['Close']
        
    final_df = final_df.join([yield_10y, emui, bci, cei, epui_m, pmi, s_df], how='left')
    
    final_df.rename(columns={
        'DGS10': 'Yield_10Y', 'WLEMUINDXD': 'EMUI',
        'BSCICP03USM665S': 'BCI', 'UMCSENT': 'CEI', 'USEPUINDXM': 'EPUI', 'INDPRO': 'PMI'
    }, inplace=True)
    
    final_df.ffill(inplace=True); final_df.bfill(inplace=True)
    
    # --- STATIONARY TRANSFORMATION (LOG-RETURNS) ---
    print("Building Stationary Log-Return features...")
    final_df['Log_Return'] = np.log(final_df['Price'] / final_df['Price'].shift(1))
    
    # Indicators on Price
    print("Adding Deep Technical Indicators (All 16 Features)...")
    final_df['EMA'] = ta.trend.ema_indicator(final_df['Price'], window=20)
    final_df['MACD_Line'] = ta.trend.macd(final_df['Price'])
    final_df['MACD_Signal'] = ta.trend.macd_signal(final_df['Price'])
    final_df['MACD_Diff'] = ta.trend.macd_diff(final_df['Price'])
    final_df['RSI'] = ta.momentum.rsi(final_df['Price'], window=14)
    final_df['Stoch_K'] = ta.momentum.stoch(final_df['High'], final_df['Low'], final_df['Price'], window=14, smooth_window=3)
    final_df['Stoch_D'] = ta.momentum.stoch_signal(final_df['High'], final_df['Low'], final_df['Price'], window=14, smooth_window=3)
    final_df['Williams_R'] = ta.momentum.williams_r(final_df['High'], final_df['Low'], final_df['Price'], lbp=14)
    
    # --- LAGGING ---
    feature_cols = [
        'Log_Return', 'EMA', 'MACD_Line', 'MACD_Signal', 'MACD_Diff', 'RSI', 
        'Stoch_K', 'Stoch_D', 'Williams_R', 'Yield_10Y', 'EMUI', 'BCI', 
        'CEI', 'EPUI', 'PMI', 'Sentiment_Score'
    ]
    
    for c in feature_cols:
        if c in final_df.columns:
            final_df[f'{c}_lag1'] = final_df[c].shift(1)
        
    final_df['Target_Return'] = final_df['Log_Return']
    
    # Final output
    final_df.dropna(inplace=True)
    # Ensure final dataset has ALL columns required
    final_df['Adj_Close'] = final_df['Price']
    final_df.to_csv(PROCESSED_FILE)
    print(f"Stationary Dataset Finalized (No Reduction): {len(final_df)} rows with {len(feature_cols)} base features.")

if __name__ == "__main__":
    main()
