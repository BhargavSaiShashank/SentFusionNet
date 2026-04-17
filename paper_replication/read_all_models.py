import joblib
import os
import pandas as pd

# Path to the Production Models folder
MODELS_DIR = r'C:\Users\shahs\FinalYear\paper_replication\models'

def read_all_production_models():
    print("--- 📚 PRODUCTION MODEL INVENTORY ---")
    
    # Iterate through all joblib files in the folder
    files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.joblib')]
    
    for filename in sorted(files):
        print(f"\n📂 FULL FILE: {filename}")
        full_path = os.path.join(MODELS_DIR, filename)
        
        try:
            # 1. Load the binary file back into a Python object 
            obj = joblib.load(full_path)
            
            # 2. Identify the type and show relevant info 
            print(f"   TYPE: {type(obj).__name__}")
            
            if "mlp" in filename:
                print(f"   DESCRIPTION: The Production Neural Network.")
                print(f"   LAYERS: {obj.hidden_layer_sizes}")
                print(f"   ITERATIONS: {obj.n_iter_}")
            
            elif "xgboost" in filename:
                print(f"   DESCRIPTION: The Production Gradient Boosting Model.")
                print(f"   ESTIMATORS: {obj.n_estimators}")
            
            elif "scaler" in filename:
                print(f"   DESCRIPTION: Data Normalization Scaler.")
                print(f"   SCALER MEAN (First 3): {obj.mean_[:3]}") 
                
            elif "rfe_selector" in filename:
                print(f"   DESCRIPTION: Feature Selection Mask.")
                # This shows which indices the selector is prioritizing
                print(f"   SELECTED FEATURES COUNT: {obj.n_features_}")
                
            elif "lr" in filename:
                print(f"   DESCRIPTION: Final Linear Regression baseline.")
                print(f"   INTERCEPT: {obj.intercept_}")

        except Exception as e:
            print(f"   ERROR READING: {e}")

if __name__ == "__main__":
    read_all_production_models()
