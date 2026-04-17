import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Paths (Fixed for Improved Branch)
ROOT_DIR = r'C:\Users\shahs\FinalYear\improved_version'
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'final_dataset.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')

def evaluate_modern_branch():
    print("--- 📊 EVALUATING MODERN BRANCH: 2020-2026 ---")
    df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    
    feature_cols = [c for c in df.columns if c.endswith('_lag1')]
    X = df[feature_cols]
    y = df['Target_Price'].values.reshape(-1, 1)
    
    # Using a Split for the "Improved" Evaluation
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Modern Scaling
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)
    
    # 1. Feature Selection (RFE)
    selector = joblib.load(os.path.join(MODELS_DIR, 'rfe_selector_production.joblib'))
    X_test_sel = selector.transform(X_test_s)
    X_train_sel = selector.transform(X_train_s)
    
    # 2. Results
    results = []
    
    # MLP (Neural Network)
    print("Benchmarking Modern MLP...")
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=2000, random_state=42)
    mlp.fit(X_train_sel, y_train_s.ravel())
    preds_s = mlp.predict(X_test_sel).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_s)
    
    results.append({
        "Model": "MLP (Modern)",
        "MSE": mean_squared_error(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    })
    
    # XGBoost
    print("Benchmarking Modern XGBoost...")
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    xgb.fit(X_train_sel, y_train_s.ravel())
    preds_s = xgb.predict(X_test_sel).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_s)
    
    results.append({
        "Model": "XGBoost (Modern)",
        "MSE": mean_squared_error(y_test, preds),
        "MAE": mean_absolute_error(y_test, preds),
        "R2": r2_score(y_test, preds)
    })
    
    res_df = pd.DataFrame(results)
    print("\n--- FINAL RESULTS (IMPROVED VERSION: 2020-2026) ---")
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    evaluate_modern_branch()
