from pdfminer.high_level import extract_text
import re
import sys

# Ensure utf-8 output to avoid Windows console encoding issues
sys.stdout.reconfigure(encoding='utf-8')

pdf_path = r'C:\Users\shahs\FinalYear\(base)SentFusionNet (2).pdf'
try:
    text = extract_text(pdf_path)
    
    # Search for metrics like RMSE, MAPE, Accuracy, DA etc.
    metrics = ["RMSE", "MAPE", "Accuracy", "MAE", "Directional", "Result", "Summary", "Performance", "Table"]
    
    print("--- PAPER CONTENT HIGHLIGHTS ---")
    for k in metrics:
        # Search for lines containing the keyword and some numbers
        matches = re.finditer(r"([^\.\n]*{}[^\.\n]*?[\d\.]+[\d\.\s\%]*)".format(k), text, re.IGNORECASE)
        count = 0
        for m in matches:
            if count > 15: break
            print(m.group(1).replace('\u2212', '-').strip())
            count += 1
            
    print("\n--- FIRST 2000 CHARS ---")
    print(text[:2000].replace('\u2212', '-'))
except Exception as e:
    print(f"Error: {e}")
