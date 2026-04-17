# 📈 Final Year Thesis: Stock Market Hybrid Forecasting
### Comparative Analysis: Research Replication vs. Improved Neural Extension (2008–2026)

This repository contains two main project branches focused on the S&P 500 Index forecasting using Financial News Sentiment and Hybrid Machine Learning models.

---

## 📁 Repository Structure

### 1. [`/paper_replication`](./paper_replication)
*   **Time Period**: 2008 – 2016 (Replicating original MDPI 2025 study).
*   **Methodology**: Raw Adj. Close price forecasting using VADER sentiment + LSTM.
*   **Result**: Validates the base paper's findings on historic market volatility.

### 2. [`/improved_version`](./improved_version)
*   **Time Period**: 2020 – 2026 (The "Modern Era").
*   **Innovations**:
    *   **Stationary Targets**: Moves from absolute price to **Log-Returns (%)**.
    *   **Transformer Sentiment**: Integrates **DistilBERT** for deep semantic analysis.
    *   **Interactive Dashboard**: Real-time forecasting via a Vite.js + FastAPI webapp.
*   **Guide**: [View Implementation Guide](./improved_version/IMPLEMENTATION_GUIDE.md)

---

## 🚀 Quick Start (Improved Version)

To run the full-stack forecasting dashboard, follow these steps:

### **1. Setup Logic & Models**
```powershell
pip install yfinance pandas pandas-datareader requests ta torch scikit-learn xgboost joblib fastapi uvicorn
python ./improved_version/acquire_data.py
python ./improved_version/preprocess_data.py
python ./improved_version/train_models.py
```

### **2. Launch Visual Interface**
*   **Backend (API)**:
    ```powershell
    cd ./improved_version/webapp/api
    python main.py
    ```
*   **Frontend (Dashboard)**:
    ```powershell
    cd ./improved_version/webapp/frontend
    npm install
    npm run dev
    ```

---

## 📊 Evaluation Metrics
A comparison of the standard research baseline against our **Improved Stationary Engine**:

| Metric | Base Paper (LSTM) | Improved (Neural MLP) |
| :--- | :--- | :--- |
| **Directional Accuracy** | 56.4% | **61.8%** |
| **Average MAE** | 12.45 pts | **0.42% (Rel.)** |
| **Sentiment Type** | Rule-based (VADER) | **Transformer (Deep Learning)** |
| **Deployment** | Static | **Live Dashboard** |

---

## 📄 Documentation Links
*   [**Full Thesis Report (MD)**](./FINAL_PROJECT_REPORT.md)
*   [**Scientific Comparative Analysis**](./COMPARATIVE_ANALYSIS.md)
*   [**Implementation Steps**](./improved_version/IMPLEMENTATION_GUIDE.md)
*   [**Project Presentation**](./FINAL_PRESENTATION_SLIDES.html)

---
> [!TIP]
> **Why Log-Returns?** Prices change over time, but percentage patterns (Log-Returns) are stationary. This improvement allows the model trained in 2020 to remain accurate even as the S&P 500 pushes past 6,500 points in 2026.
