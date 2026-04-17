import pandas as pd
import numpy as np
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt

# Paths
ROOT_DIR = r'C:\Users\shahs\FinalYear\paper_replication'
DATA_FILE = os.path.join(ROOT_DIR, 'data', 'final_dataset.csv')
RESULTS_DIR = os.path.join(ROOT_DIR, 'results')

# SentFusionNet Style Hybrid Model: Gated Sentiment Fusion
class HybridSentFusion(nn.Module):
    def __init__(self, time_series_size, sentiment_size, hidden_size=64):
        super(HybridSentFusion, self).__init__()
        
        # Branch 1: LSTM for Time-Series (Technical indicators + Macro)
        self.lstm = nn.LSTM(time_series_size, hidden_size, num_layers=2, batch_first=True)
        
        # Branch 2: MLP for Sentiment / Macro snapshot
        self.sentiment_mlp = nn.Sequential(
            nn.Linear(sentiment_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, hidden_size)
        )
        
        # Gating Mechanism (Gated Fusion Unit)
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * 2, 1),
            nn.Sigmoid()
        )
        
        # Final Regression layers
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x_ts, x_sent):
        # x_ts: (batch, seq, ts_size)
        # x_sent: (batch, sent_size)
        
        # 1. Process Time-Series
        lstm_out, _ = self.lstm(x_ts)
        h_ts = lstm_out[:, -1, :] # Last hidden state (batch, hidden_size)
        
        # 2. Process Sentiment/Macro Snapshot
        h_sent = self.sentiment_mlp(x_sent) # (batch, hidden_size)
        
        # 3. Concatenate and apply Gated Fusion (Optional, standard is simple concat + MLP)
        # We'll use a strong concat + MLP for now as it's the most stable "Hybrid"
        combined = torch.cat((h_ts, h_sent), dim=1) # (batch, hidden_size * 2)
        
        # Gating (Optional)
        g = self.gate(combined)
        fused = combined * g # Weighting the combined representation
        
        out = self.fc(combined) # Predicting
        return out

def load_data_v2():
    df = pd.read_csv(DATA_FILE, index_col='Date', parse_dates=True)
    
    # 1. Time Series Features (Technicals + Returns)
    ts_cols = ['Close', 'RSI', 'MACD', 'SMA_20', 'SMA_50', 'EMA_20', 'Volatility']
    # 2. Sentiment + Macro snapshot Features
    sent_cols = ['Sentiment_Score', 'Yield_10Y', 'EPUI', 'EMUI', 'BCI', 'CEI', 'PMI']
    
    X_ts = df[ts_cols]
    X_sent = df[sent_cols]
    y = df['Target_Next_Return']
    
    split_idx = int(len(df) * 0.8)
    
    # Scale separately?
    scaler_ts = StandardScaler()
    scaler_sent = StandardScaler()
    
    X_ts_scaled = scaler_ts.fit_transform(X_ts)
    X_sent_scaled = scaler_sent.fit_transform(X_sent)
    
    return X_ts_scaled, X_sent_scaled, y.values, split_idx

def create_hybrid_seq(X_ts, X_sent, y, seq_length=10):
    xs_ts, xs_sent, ys = [], [], []
    for i in range(len(X_ts) - seq_length):
        xs_ts.append(X_ts[i:i+seq_length])
        # We take the sentiment info of the LAST day of the sequence
        xs_sent.append(X_sent[i+seq_length-1]) 
        ys.append(y[i+seq_length])
    return np.array(xs_ts), np.array(xs_sent), np.array(ys)

def run_hybrid():
    print("Initializing Hybrid SentFusionNet...")
    X_ts, X_sent, y, split_idx = load_data_v2()
    
    seq_length = 10
    X_ts_train, X_sent_train, y_train = create_hybrid_seq(X_ts[:split_idx], X_sent[:split_idx], y[:split_idx], seq_length)
    X_ts_test, X_sent_test, y_test = create_hybrid_seq(X_ts[split_idx:], X_sent[split_idx:], y[split_idx:], seq_length)
    
    # Tensors
    X_ts_train_t = torch.tensor(X_ts_train, dtype=torch.float32)
    X_sent_train_t = torch.tensor(X_sent_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).reshape(-1, 1)
    
    X_ts_test_t = torch.tensor(X_ts_test, dtype=torch.float32)
    X_sent_test_t = torch.tensor(X_sent_test, dtype=torch.float32)
    
    dataset = TensorDataset(X_ts_train_t, X_sent_train_t, y_train_t)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    model = HybridSentFusion(time_series_size=X_ts_train.shape[2], sentiment_size=X_sent_train.shape[1])
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    print("Training Hybrid Model...")
    epochs = 30
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0
        for bx_ts, bx_sent, by in loader:
            optimizer.zero_grad()
            out = model(bx_ts, bx_sent)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {epoch_loss/len(loader):.6f}")

    model.eval()
    with torch.no_grad():
        preds_t = model(X_ts_test_t, X_sent_test_t)
        preds = preds_t.numpy().flatten()
    
    # Metrics
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test, preds)
    acc = np.mean(np.sign(y_test) == np.sign(preds))
    
    metrics = {"RMSE": rmse, "MAPE": mape, "DA": acc}
    print(f"\nHybrid Results: {metrics}")
    return metrics, preds, y_test

def main():
    metrics, preds, actual = run_hybrid()
    
    # Load previous comparison
    comp_file = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    if os.path.exists(comp_file):
        comp_df = pd.read_csv(comp_file)
    else:
        comp_df = pd.DataFrame(columns=["Model", "RMSE", "MAPE", "DA"])
    
    # Remove existing Hybrid if any
    comp_df = comp_df[comp_df['Model'] != 'Hybrid']
    
    # Append Hybrid results
    new_res = pd.DataFrame([{"Model": "Hybrid", **metrics}])
    comp_df = pd.concat([comp_df, new_res], ignore_index=True)
    comp_df.to_csv(comp_file, index=False)
    print(f"Updated results saved to {comp_file}")
    
    # Plotting comparison
    plt.figure(figsize=(10, 6))
    plt.plot(actual[:50], label='Actual', alpha=0.8)
    plt.plot(preds[:50], label='Hybrid Predictions', alpha=0.8)
    plt.title('Hybrid Model: Next Return Prediction (First 50 Days of Test)')
    plt.legend()
    plt.savefig(os.path.join(RESULTS_DIR, 'hybrid_v_actual.png'))
    print(f"Plot saved to {os.path.join(RESULTS_DIR, 'hybrid_v_actual.png')}")

if __name__ == "__main__":
    main()
