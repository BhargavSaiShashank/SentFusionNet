import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score
import joblib

# Paths (Fixed for Stationary Branch)
ROOT_DIR = r'C:\Users\shahs\FinalYear\improved_version'
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'final_dataset.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import joblib

# Deep Learning Model Definition
class DeepForecaster(nn.Module):
    def __init__(self, input_size):
        super(DeepForecaster, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.network(x)

def train_stationary_models():
    print("--- DEEP LEARNING PRODUCTION TRAINING (NO REDUCTION) ---")
    df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    
    # Using ALL lagged features (No reduction)
    feature_names = [c for c in df.columns if c.endswith('_lag1')]
    print(f"Using Deep Features: {feature_names}")
    
    X = df[feature_names].values
    y = df['Target_Return'].values.reshape(-1, 1)
    
    # SCALING
    scaler_x = StandardScaler()
    X_s = scaler_x.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_s = scaler_y.fit_transform(y)
    
    joblib.dump(scaler_x, os.path.join(MODELS_DIR, 'scaler_x_production.joblib'))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, 'scaler_y_production.joblib'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names.joblib'))
    
    # PyTorch Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X_s, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_s, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = DeepForecaster(input_size=len(feature_names)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training Deep Neural Network on {len(df)} rows...")
    epochs = 100
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss/len(loader):.6f}")
            
    # Save the PyTorch Model
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'deep_model_weights.pth'))
    
    # Also save a dummy MLP for compatibility with any legacy scripts if needed, 
    # but the primary is now the PyTorch model.
    print("Deep Learning Training Complete. No feature reduction applied.")

if __name__ == "__main__":
    train_stationary_models()
