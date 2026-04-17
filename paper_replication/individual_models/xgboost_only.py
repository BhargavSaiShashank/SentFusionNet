import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Paths
DATA_FILE = r'C:\Users\shahs\FinalYear\paper_replication\data\final_dataset.csv'

def run_xgboost():
    print("--- 🧠 STANDALONE XGBOOST (GRADIENT BOOSTING) ---")
    df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    
    # Selecting the top 10 features from Scenario C
    feature_cols = ['Adj_Close_lag1', 'EMA_lag1', 'MACD_Line_lag1', 'MACD_Signal_lag1', 'MACD_Diff_lag1', 'RSI_lag1', 'Yield_10Y_lag1', 'EMUI_lag1', 'BCI_lag1', 'Sentiment_Score_lag1']
    
    X = df[feature_cols]
    y = df['Target_Price'].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Scaling X and y manually to match paper
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)
    
    # 100% Replication XGBoost 
    model = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    model.fit(X_train_s, y_train_s.ravel())
    
    # Prediction and scaling back to price levels
    preds_s = model.predict(X_test_s).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_s)
    
    print(f"Results for XGBoost (Scenario C):")
    print(f"MSE: {mean_squared_error(y_test, preds):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"R2: {r2_score(y_test, preds):.4f}")

if __name__ == "__main__":
    run_xgboost()
