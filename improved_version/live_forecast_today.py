import yfinance as yf
import pandas as pd
import joblib
import os
import ta
import torch
from transformers import pipeline
import warnings
warnings.filterwarnings('ignore')

# Paths
ROOT_DIR = r'C:\Users\shahs\FinalYear\improved_version'
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

def get_live_forecast():
    print("--- 🔮 LIVE FORECAST: TODAY (APRIL 1, 2026) ---")
    
    # 1. Loading the calibrated modern models
    print("Loading modern predictive brains...")
    scaler_x = joblib.load(os.path.join(MODELS_DIR, 'scaler_x_production.joblib'))
    scaler_y = joblib.load(os.path.join(MODELS_DIR, 'scaler_y_production.joblib'))
    selector = joblib.load(os.path.join(MODELS_DIR, 'rfe_selector_production.joblib'))
    mlp = joblib.load(os.path.join(MODELS_DIR, 'mlp_production.joblib'))
    
    # 2. Fetching Recent Market Data
    print("Fetching recent market data (60-day window)...")
    sp500 = yf.download('^GSPC', period='60d') 
    
    # Pre-processing Technicals
    df = sp500.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    df['Adj_Close'] = df['Close']
    df['EMA'] = ta.trend.ema_indicator(df['Adj_Close'], window=20)
    df['MACD_Line'] = ta.trend.macd(df['Adj_Close'])
    df['MACD_Signal'] = ta.trend.macd_signal(df['Adj_Close'])
    df['MACD_Diff'] = ta.trend.macd_diff(df['Adj_Close'])
    df['RSI'] = ta.momentum.rsi(df['Adj_Close'], window=14)
    df['Stoch_K'] = ta.momentum.stoch(df['High'], df['Low'], df['Adj_Close'], window=14)
    df['Stoch_D'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Adj_Close'], window=14)
    df['Williams_R'] = ta.momentum.williams_r(df['High'], df['Low'], df['Adj_Close'], lbp=14)
    
    # 3. Last Known Macro (Modern proxies)
    df['Yield_10Y'] = 4.2 
    df['EMUI'] = 100.0
    df['BCI'] = 100.0
    df['CEI'] = 100.0
    df['EPUI'] = 100.0
    df['PMI'] = 50.0
    
    # 4. Scrape Yesterday's News Headlines for Today's Sentiment
    print("Scraping transformer-based sentiment from Yahoo Finance News...")
    ticker = yf.Ticker('^GSPC')
    # Filter for titles only
    headlines_raw = ticker.news[:10]
    headlines = [n['title'] for n in headlines_raw if 'title' in n]
    # Fallback to content key if title not at top level
    if not headlines:
        headlines = [n['content']['title'] for n in headlines_raw if 'content' in n]
        
    combined_news = '. '.join(headlines)
    
    device = 0 if torch.cuda.is_available() else -1
    sentiment_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english", device=device)
    res = sentiment_pipeline(combined_news[:512])[0]
    score = res['score']
    if res['label'] == 'NEGATIVE': score = -score
    
    df['Sentiment_Score'] = score
    
    # 5. Build Final Feature Row using ILOC (Yesterday T-1)
    # The models expect: ['Adj_Close_lag1', 'EMA_lag1'...] 
    feature_names = ['Adj_Close_lag1', 'EMA_lag1', 'MACD_Line_lag1', 'MACD_Signal_lag1', 'MACD_Diff_lag1', 'RSI_lag1', 'Stoch_K_lag1', 'Stoch_D_lag1', 'Williams_R_lag1', 'Yield_10Y_lag1', 'EMUI_lag1', 'BCI_lag1', 'CEI_lag1', 'EPUI_lag1', 'PMI_lag1', 'Sentiment_Score_lag1']
    
    # Grab the very last row (Yesterday) and treat as the lagged input
    last_row = df.iloc[-1]
    last_date = df.index[-1].date()
    
    input_values = [
        last_row['Adj_Close'], last_row['EMA'], last_row['MACD_Line'], 
        last_row['MACD_Signal'], last_row['MACD_Diff'], last_row['RSI'], 
        last_row['Stoch_K'], last_row['Stoch_D'], last_row['Williams_R'],
        last_row['Yield_10Y'], last_row['EMUI'], last_row['BCI'],
        last_row['CEI'], last_row['EPUI'], last_row['PMI'],
        last_row['Sentiment_Score']
    ]
    
    input_df = pd.DataFrame([input_values], columns=feature_names)
    
    # 6. Execute Production Predictor
    input_s = scaler_x.transform(input_df)
    input_sel = selector.transform(input_s)
    pred_s = mlp.predict(input_sel).reshape(-1, 1)
    prediction = scaler_y.inverse_transform(pred_s)[0][0]
    
    print("\n--- SUMMARY: APRIL 1, 2026 ---")
    print(f"Yesterday's Close ({last_date}): {last_row['Adj_Close']:.2f}")
    print(f"Modern Sentiment Polarization: {score:.4f} ({res['label']})")
    print("-" * 30)
    print(f"🚨 PROJECTED S&P 500 PRICE FOR TODAY: {prediction:.2f}")
    print("-" * 30)
    
    diff = prediction - last_row['Adj_Close']
    signal = "⬆️ BULLISH (Profit Taking Zone)" if diff > 0 else "⬇️ BEARISH (Resistance Zone)"
    print(f"DIRECTIONAL SIGNAL: {signal}")
    print(f"ESTIMATED CHANGE: {diff:+.2f} points")

if __name__ == "__main__":
    get_live_forecast()
