import sys
from pdfminer.high_level import extract_text
sys.stdout.reconfigure(encoding='utf-8')

pdf_path = r'C:\Users\shahs\FinalYear\(base)SentFusionNet (2).pdf'
try:
    text = extract_text(pdf_path)
    # Search for all mentions of Tables and their content
    import re
    tables = re.finditer(r"(Table\s+\d+.*?)(?=Table\s+\d+|$)", text, re.DOTALL | re.IGNORECASE)
    for t in tables:
        content = t.group(1).replace('\u2212', '-')
        # If it looks like a result table (contains Error or MSE or MAE or RMSE)
        if any(kw in content.upper() for kw in ["MSE", "MAE", "RMSE", "ACCURACY", "R2", "LR", "MLP", "XGBOOST"]):
            print("--- TABLE FOUND ---")
            print(content[:1500]) # Print first 1500 chars of each interesting table
            print("-" * 20)
except Exception as e:
    print(f"Error: {e}")
