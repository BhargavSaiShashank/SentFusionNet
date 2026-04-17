import joblib
import pandas as pd
import os

# Paths to models
MODELS_DIR = r'C:\Users\shahs\FinalYear\paper_replication\models'
MODEL_PATH = os.path.join(MODELS_DIR, 'mlp_selection_final.joblib')
XGB_PATH = os.path.join(MODELS_DIR, 'xgb_selection_final.joblib')

def inspect_model():
    print("--- 🧠 INSPECTING MODEL BRAINS ---")
    
    # 1. Inspecting XGBoost (Most readable)
    if os.path.exists(XGB_PATH):
        print("\n[Reading XGBoost Knowledge...]")
        model = joblib.load(XGB_PATH)
        # This shows how much 'weight' the model gives to each input
        importances = model.feature_importances_
        # Since we use 10 features for Scenario C
        feature_names = ['Adj_Close_lag1', 'EMA_lag1', 'MACD_Line_lag1', 'MACD_Signal_lag1', 'MACD_Diff_lag1', 'RSI_lag1', 'Yield_10Y_lag1', 'EMUI_lag1', 'BCI_lag1', 'Sentiment_Score_lag1']
        
        feat_df = pd.DataFrame({'Input': feature_names, 'Importance (%)': importances * 100})
        print(feat_df.sort_values(by='Importance (%)', ascending=False))

    # 2. Inspecting MLP (Neural Network)
    if os.path.exists(MODEL_PATH):
        print("\n[Reading MLP (Neural Network) Architecture...]")
        model = joblib.load(MODEL_PATH)
        print(f"Number of layers: {model.n_layers_}")
        print(f"Layer sizes: {model.hidden_layer_sizes}")
        print(f"Learning Rate: {model.learning_rate_init}")
        print(f"Final training loss: {model.loss_}")

if __name__ == "__main__":
    inspect_model()
