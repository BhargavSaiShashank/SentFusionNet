import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import joblib

# Paths (Localized for India)
ROOT_DIR = r'C:\Users\shahs\FinalYear\improved_version\india'
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'india_final_dataset.csv')
MODELS_DIR = os.path.join(ROOT_DIR, 'models')
os.makedirs(MODELS_DIR, exist_ok=True)

# Deep Learning Model Definition (Must match the US version for consistency)
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

def train_india_models():
    print("--- INDIAN DEEP LEARNING TRAINING (NIFTY 50) ---")
    df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)
    
    # Selecting all lagged features
    feature_names = [c for c in df.columns if c.endswith('_lag1')]
    print(f"Features: {feature_names}")
    
    X = df[feature_names].values
    y = df['Target_Return'].values.reshape(-1, 1)
    
    # Scaling
    scaler_x = StandardScaler()
    X_s = scaler_x.fit_transform(X)
    
    scaler_y = StandardScaler()
    y_s = scaler_y.fit_transform(y)
    
    joblib.dump(scaler_x, os.path.join(MODELS_DIR, 'scaler_x_india.joblib'))
    joblib.dump(scaler_y, os.path.join(MODELS_DIR, 'scaler_y_india.joblib'))
    joblib.dump(feature_names, os.path.join(MODELS_DIR, 'feature_names_india.joblib'))
    
    # PyTorch Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    X_t = torch.tensor(X_s, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_s, dtype=torch.float32).to(device)
    
    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = DeepForecaster(input_size=len(feature_names)).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print(f"Training on {len(df)} rows of Nifty 50 data...")
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
    torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'india_model_weights.pth'))
    print("Indian Deep Learning Training Complete.")

if __name__ == "__main__":
    train_india_models()
