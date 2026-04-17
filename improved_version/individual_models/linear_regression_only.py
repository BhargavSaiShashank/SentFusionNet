import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Paths
DATA_FILE = r'C:\Users\shahs\FinalYear\paper_replication\data\final_dataset.csv'

def run_linear_regression():
    print("--- 🧠 STANDALONE LINEAR REGRESSION ---")
    df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    
    # Paper-identified Top 10 features
    feature_cols = ['Adj_Close_lag1', 'EMA_lag1', 'MACD_Line_lag1', 'MACD_Signal_lag1', 'MACD_Diff_lag1', 'RSI_lag1', 'Yield_10Y_lag1', 'EMUI_lag1', 'BCI_lag1', 'Sentiment_Score_lag1']
    
    X = df[feature_cols]
    y = df['Target_Price'].values.reshape(-1, 1)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)
    
    # Scaling X and y manually
    scaler_x = StandardScaler()
    X_train_s = scaler_x.fit_transform(X_train)
    X_test_s = scaler_x.transform(X_test)
    
    scaler_y = StandardScaler()
    y_train_s = scaler_y.fit_transform(y_train)
    y_test_s = scaler_y.transform(y_test)
    
    # Classic Linear Regression (Base model for both paper and replication)
    model = LinearRegression()
    model.fit(X_train_s, y_train_s.ravel())
    
    # Prediction and price conversion
    preds_s = model.predict(X_test_s).reshape(-1, 1)
    preds = scaler_y.inverse_transform(preds_s)
    
    print(f"Results for Linear Regression (Scenario C):")
    print(f"MSE: {mean_squared_error(y_test, preds):.2f}")
    print(f"MAE: {mean_absolute_error(y_test, preds):.2f}")
    print(f"R2: {r2_score(y_test, preds):.4f}")

if __name__ == "__main__":
    run_linear_regression()
