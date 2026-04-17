import pandas as pd
import numpy as np
import os
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error

# Paths
ROOT_DIR = r'C:\Users\shahs\FinalYear\improved_version'
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'final_dataset.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

def perform_backtest():
    print("--- 📉 PERFORMING FULL BACKTEST (2024-2026) ---")
    
    # 1. Loading data and models
    df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    scaler_x = joblib.load(os.path.join(MODELS_DIR, 'scaler_x_production.joblib'))
    scaler_y = joblib.load(os.path.join(MODELS_DIR, 'scaler_y_production.joblib'))
    mlp = joblib.load(os.path.join(MODELS_DIR, 'mlp_production.joblib'))
    
    # Selecting modern period (2024 onwards)
    test_df = df[df.index >= '2024-01-01'].copy()
    
    # Feature Names (Must match training)
    feature_names = ['Log_Return_lag1', 'EMA_lag1', 'MACD_Line_lag1', 'RSI_lag1', 'Yield_10Y_lag1', 'Sentiment_Score_lag1']
    X = test_df[feature_names]
    
    # 2. Iterative Prediction
    X_s = scaler_x.transform(X)
    pred_s = mlp.predict(X_s).reshape(-1, 1)
    pred_log_ret = scaler_y.inverse_transform(pred_s).flatten()
    
    # Convert Log Returns back to Prices
    # Price(t) = Price(t-1) * exp(Log_Return)
    # Since features were lagged, the predicted price is for Current Date
    actual_prices = test_df['Adj_Close'].values
    
    # To get "True" historical predictions, we use the price from the PREVIOUS row
    # Because we are predicting P(t) using X(t-1)
    # We shift the pred_log_ret by 1 to align with original prices correctly
    predicted_prices = test_df['Adj_Close'].shift(1).values * np.exp(pred_log_ret)
    
    # 3. Validation Metrics
    # Dropping the first NaN row from the shift
    valid_mask = ~np.isnan(predicted_prices)
    y_true = actual_prices[valid_mask]
    y_pred = predicted_prices[valid_mask]
    
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    
    print(f"Backtest R2: {r2:.6f}")
    print(f"Backtest MAE: {mae:.2f} points")
    
    # 4. Generate High-Fidelity Plot
    plt.figure(figsize=(12, 6), dpi=150)
    plt.style.use('dark_background')
    
    plt.plot(test_df.index[valid_mask], y_true, label='Actual S&P 500', color='#06b6d4', linewidth=2, alpha=0.8)
    plt.plot(test_df.index[valid_mask], y_pred, label='Neural Forecast', color='#f43f5e', linestyle='--', linewidth=1.5, alpha=0.9)
    
    plt.title(f'S&P 500 Neural Backtest (2024-2026)\nR2: {r2:.4f} | Avg Error: {mae:.2f} pts', fontsize=12, pad=20)
    plt.ylabel('Price (USD)')
    plt.xlabel('Market Date')
    plt.legend()
    plt.grid(color='white', alpha=0.1)
    plt.tight_layout()
    
    plot_path = os.path.join(RESULTS_DIR, 'full_backtest_proof.png')
    plt.savefig(plot_path)
    print(f"✅ FINAL PROOF GENERATED: {plot_path}")
    
if __name__ == "__main__":
    perform_backtest()
