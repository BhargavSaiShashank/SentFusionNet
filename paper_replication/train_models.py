import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE
import joblib

# Paths
ROOT_DIR = r'C:\Users\shahs\FinalYear\paper_replication'
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'final_dataset.csv')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

def train_production_models():
    print("--- 🚀 FINAL PRODUCTION TRAINING (100% DATA) ---")
    df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    
    feature_cols = [c for c in df.columns if c.endswith('_lag1')]
    X = df[feature_cols]
    y = df['Target_Price'].values.reshape(-1, 1)
    
    print(f"Training on the FULL dataset: {len(df)} rows.")

    # 1. Feature Scaling (Using 100% data)
    scaler_x = StandardScaler()
    X_s = scaler_x.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_s = scaler_y.fit_transform(y)
    
    # Save scalers for production use
    joblib.dump(scaler_x, os.path.join(MODELS_DIR, 'scaler_x_production.joblib'))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, 'scaler_y_production.joblib'))
    
    # 2. FEATURE SELECTION (RFE on Full Data)
    print("Executing final Feature Selection (RFE)...")
    estimator = RandomForestRegressor(n_estimators=50, random_state=42)
    selector = RFE(estimator, n_features_to_select=10, step=1)
    selector = selector.fit(X_s, y_s.ravel())
    X_sel = selector.transform(X_s)
    
    selected_features = [f for f, s in zip(feature_cols, selector.support_) if s]
    print(f"Final production top features: {selected_features}")
    joblib.dump(selector, os.path.join(MODELS_DIR, 'rfe_selector_production.joblib'))
    
    # 3. TRAINING FINAL ARCHITECTURES
    
    # Linear Regression
    print("Training Linear Regression (Production)...")
    lr = LinearRegression()
    lr.fit(X_sel, y_s.ravel())
    joblib.dump(lr, os.path.join(MODELS_DIR, 'lr_production.joblib'))
    
    # XGBoost
    print("Training XGBoost (Production)...")
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42)
    xgb.fit(X_sel, y_s.ravel())
    joblib.dump(xgb, os.path.join(MODELS_DIR, 'xgboost_production.joblib'))
    
    # MLP (Neural Network)
    print("Training MLP Neural Network (Production)...")
    # Paper-style refined architecture (100, 50, 25)
    mlp = MLPRegressor(hidden_layer_sizes=(100, 50, 25), max_iter=2000, random_state=42)
    mlp.fit(X_sel, y_s.ravel())
    joblib.dump(mlp, os.path.join(MODELS_DIR, 'mlp_production.joblib'))
    
    print("\n✅ All Production Models Trained and Saved to /models/.")

if __name__ == "__main__":
    train_production_models()
