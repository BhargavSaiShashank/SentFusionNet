# 🚀 Improved Version Implementation Guide
### Hybrid Stock Forecasting & Web Dashboard

This document outlines the step-by-step implementation and execution process for the **Improved Version** of the Stock Forecasting Thesis Project. This version transitions from raw price prediction to **Stationary Log-Return Forecasting** with a real-time Vite.js dashboard.

---

## 📋 Table of Contents
1. [Prerequisites](#1-prerequisites)
2. [Data Pipeline](#2-data-pipeline)
3. [Model Pipeline](#3-model-pipeline)
4. [Web Application (Launch)](#4-web-application-launch)
5. [Evaluation & Backtesting](#5-evaluation--backtesting)

---

## 1. Prerequisites
Ensure you have Python 3.9+ and Node.js installed. Run the following command to install all necessary Python libraries:

```powershell
pip install yfinance pandas pandas-datareader requests ta torch scikit-learn xgboost joblib fastapi uvicorn
```

---

## 2. Data Pipeline
The improved version fetches modern data (2020-2026) and performs stationary transformations to prevent the "Mean Reversion Trap" in price-based prediction.

### **Step A: Acquire Raw Data**
Downloads S&P 500 (^GSPC) and Macroeconomic indicators (DGS10, EMUI, BCI, etc.) from FRED.
```powershell
python C:\Users\shahs\FinalYear\improved_version\acquire_data.py
```

### **Step B: Preprocessing**
Merges datasets, handles missing values, computes **Log-Returns**, adds Technical Indicators, and generates a time-lagged `final_dataset.csv`.
```powershell
python C:\Users\shahs\FinalYear\improved_version\preprocess_data.py
```

---

## 3. Model Pipeline
Trains the production-grade MLP (Multi-Layer Perceptron) Neural Network.

```powershell
# Trains the logic and saves .joblib files to /models/
python C:\Users\shahs\FinalYear\improved_version\train_models.py
```

---

## 4. Web Application (Launch)
The dashboard requires both the FastAPI backend and the Vite.js frontend to be running simultaneously.

### **Backend: FastAPI Server**
The backend serves predictions via the `/api/forecast` endpoint on port **8001**.
```powershell
cd C:\Users\shahs\FinalYear\improved_version\webapp\api
python main.py
```

### **Frontend: React + Vite Dashboard**
The frontend displays the live chart, sentiment signals, and feature importance.
```powershell
cd C:\Users\shahs\FinalYear\improved_version\webapp\frontend
npm install
npm run dev
```
👉 *Open [http://localhost:5173](http://localhost:5173) in your browser.*

---

## 5. Evaluation & Backtesting
To verify the performance of the Improved Version against the base paper benchmarks:

| Script | Purpose |
| :--- | :--- |
| `evaluate_modern.py` | Shows MAE, RMSE, and Directional Accuracy for the 2020-2026 period. |
| `run_backtest.py` | Runs a trading simulation to compare ROI against a Buy-and-Hold strategy. |
| `hybrid_model.py` | Implementation of the Combined MLP + XGBoost stacking logic. |

```powershell
python C:\Users\shahs\FinalYear\improved_version\evaluate_modern.py
python C:\Users\shahs\FinalYear\improved_version\run_backtest.py
```

---
> [!IMPORTANT]
> **Stationary Note**: This version predicts **percentage changes** (Log-Returns), not raw prices. This ensures the model learns actual volatility patterns rather than just following a trend.
