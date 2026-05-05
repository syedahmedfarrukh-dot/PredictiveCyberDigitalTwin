# 🔷 Predictive Cyber Digital Twin

**CS-333: Applied AI & Machine Learning — Capstone Project**

| | |
|---|---|
| **Group** | Predictive Cyber Digital Twin |
| **Members** | Tooba Riaz · Muhammad Hassaan Khan · Anees Ahmed · Syed Ahmed Farukh M |
| **Domain** | Predictive Maintenance · Industrial IoT · Time-Series ML |

---

## 📌 Problem Definition

Modern industrial machines generate continuous sensor data — but traditional monitoring only reacts *after* a failure. This project builds a **Predictive Cyber Digital Twin** that creates a virtual model of a motor/pump system, monitors 5 real-time sensor readings, and uses ML to **predict failures before they happen**.

**Why AI/ML?** Physics equations simulate current state, but cannot predict future degradation from noisy time-series. Random Forest learns non-linear patterns across rolling sensor windows that deterministic models miss.

---

## 🗂 Project Structure

```
cyber-twin/
├── data_generator.py      # Module 1: Synthetic sensor data + fault injection
├── digital_twin.py        # Module 2: Physics-based virtual machine model
├── ml_predictor.py        # Module 3: Random Forest prediction engine
├── main.py                # Module 4: Orchestrator + interactive health report
├── eda.py                 # EDA: Exploratory data analysis (5 charts)
├── baseline_model.py      # Baseline: Linear Regression vs Random Forest
├── explainability.py      # XAI: Feature importance + SHAP + residuals
├── ethics.py              # Ethics: Bias, fairness, privacy, deployment risks
├── machine_data.csv       # Dataset: 500-sample time-series sensor data
├── requirements.txt       # Python dependencies
└── README.md
```

---

## 📊 Dataset

| Property | Value |
|---|---|
| Size | 500 samples × 6 columns |
| Type | Synthetic time-series (physics-based) |
| Columns | time, rpm, temperature, vibration, load, pressure |
| Fault Events | Overheating (t=150–170), Bearing Wear (t=320–340), Load Surge (t=420–440) |

---

## 🤖 Model Results

| Model | Target | MAE | R² |
|---|---|---|---|
| Linear Regression (Baseline) | Temperature | 3.063 | 0.803 |
| **Random Forest (Advanced)** | Temperature | **2.823** | **0.833** |
| Linear Regression (Baseline) | Vibration | 0.308 | 0.701 |
| **Random Forest (Advanced)** | Vibration | **0.300** | **0.698** |

---

## 🚀 How to Run

```bash
pip install -r requirements.txt

python eda.py              # Exploratory Data Analysis
python baseline_model.py   # Baseline vs Advanced comparison
python explainability.py   # Feature importance + SHAP
python ethics.py           # Ethical AI analysis
python main.py             # Full interactive system
```

---

## 📚 References

1. Grieves, M. (2014). Digital Twin: Manufacturing Excellence through Virtual Factory Replication.
2. Breiman, L. (2001). Random Forests. Machine Learning, 45, 5–32.
3. Lundberg & Lee (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
4. ISO 13374:2003 — Condition monitoring and diagnostics of machines.
