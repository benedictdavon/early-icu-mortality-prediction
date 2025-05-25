# ICU Mortality Prediction (AI Capstone 2025)

Predicting in-hospital mortality using vital signs, lab results, and demographic data collected within the first 6 hours of ICU admission.  
Final project for the **NYCU AI Capstone 2025** course, based on the **MIMIC-IV** dataset.

---

## 📌 Overview

This project develops machine learning models to predict ICU patient mortality using early-stage clinical data. The goal is to provide clinicians with reliable risk assessment tools to inform critical care decisions.

**Key Achievements:**

- 🎯 **Best Performance**: XGBoost ensemble (AUC: 0.8465, F1: 0.5060)
- 📊 **Dataset**: 16,922 ICU patients (20.87% mortality rate)
- ⚡ **GPU Acceleration**: CUDA-optimized XGBoost training
- 🔬 **Clinical Focus**: Medical domain knowledge throughout pipeline

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/benedictdavon/early-icu-mortality-prediction
cd early-icu-mortality-prediction

# Setup environment
conda env create -f environment.yml
conda activate icu-mortality-prediction
```

### Run Models

```bash
# Best performing model
python src/main.py --model xgboost_ensemble --ensemble-size 7

# Individual models
python src/main.py --model xgboost --tune
python src/main.py --model random_forest --tune
python src/main.py --model logistic_regression --tune
```

---

## 📈 Results Summary

| Model                      | AUC        | Accuracy   | Precision  | Recall     | F1-Score   |
| -------------------------- | ---------- | ---------- | ---------- | ---------- | ---------- |
| **XGBoost Ensemble (n=7)** | **0.8465** | **82.98%** | **64.19%** | **41.64%** | **0.5052** |
| XGBoost Single             | 0.8426     | 80.95%     | 53.70%     | 62.75%     | 0.5787     |
| Random Forest              | 0.8145     | 79.00%     | 49.71%     | 60.91%     | 0.5474     |
| Logistic Regression        | 0.7876     | 74.39%     | 42.44%     | 64.02%     | 0.5104     |

**Clinical Thresholds:**

- **Standard (0.50)**: High precision for avoiding false alarms
- **F1-optimized (0.45)**: Best balance for general clinical use
- **Balanced (0.30)**: Good for screening applications
- **High-sensitivity (0.10)**: Maximum recall for critical care

---

## 🏗️ Architecture

```
Data Pipeline: MIMIC-IV → Cohort Selection → Feature Extraction → Preprocessing → Models
```

**Key Components:**

- **469 raw features** → **156 optimized features** (XGBoost)
- **Advanced preprocessing**: MICE imputation, clinical outlier detection
- **Model-specific optimization**: Tailored feature selection per algorithm
- **Ensemble methodology**: 5-10 models with prediction averaging

---

## 📂 Repository Structure

```bash
├── src/
│   ├── main.py              # Model training & evaluation
│   ├── models/              # XGBoost, Random Forest, Logistic Regression
│   ├── preprocessing/       # Advanced data preprocessing modules
│   └── feature_extraction/  # Clinical feature engineering
├── data/processed/          # Model-ready datasets
├── results/                 # Model outputs & visualizations
├── notebooks/               # Analysis & exploration
└── requirements.txt         # Dependencies
```

---

## 🔬 Key Features

**Clinical Integration:**

- Medical domain knowledge in feature engineering
- SIRS criteria, shock index, critical value detection
- Multiple threshold strategies for different care scenarios

**Technical Innovation:**

- GPU-accelerated XGBoost with ensemble methods
- Advanced missing data handling (MICE, KNN imputation)
- Comprehensive model interpretation (SHAP, permutation importance)

**Robust Evaluation:**

- 5-fold stratified cross-validation
- Multiple threshold analysis for clinical utility
- Feature importance analysis across all models

---

## 👨‍🏫 Team & Course

**Student:** Benedict Davon Martono 周恭麟 (110550201)  
**Course:** AI in EHR – AI Capstone 2025, NYCU  
**Instructor:** Prof. 王才沛

**Original Group Collaboration:**

- Tymofii Voitekh 提姆西
- Jorge Tyrakowski 狄豪飛

_This repository reflects my personal implementation and exploration of the project._

---

## 📖 Documentation

For detailed technical documentation, see [`REPORT.md`](report/REPORT.md):

- Complete pipeline methodology
- Comprehensive results analysis
- Feature importance interpretations
- Clinical implications and future work

---

## 📚 References

- MIMIC-IV Dataset: https://mimic.mit.edu/
- Course Materials: Provided by NYCU
- [Complete reference list in REPORT.md](report/REPORT.md#references)

---

**License:** Academic use only | **Data:** MIMIC-IV (PhysioNet Credentialed Health Data License)
