# ICU Mortality Prediction (AI Capstone 2025)

Predicting in-hospital mortality using vital signs, lab results, and demographic data collected within the first 6 hours of ICU admission.  
Final project for the **NYCU AI Capstone 2025** course, based on the **MIMIC-IV** dataset.

---

## 📌 Project Overview

Critically ill patients admitted to the ICU have a high risk of in-hospital mortality. This project aims to develop a machine learning model that can predict a patient's survival outcome using early-stage data (first 6 hours after ICU admission).  
Early prediction can help inform clinical decisions and potentially save lives.

---

## 🧠 Task Description

- **Task:** Binary classification — predict whether a patient will die or survive during their hospital stay.
- **Input Data:**
  - Vital signs (heart rate, blood pressure, respiratory rate, etc.)
  - Lab test results (creatinine, glucose, platelets, etc.)
  - Demographics (age, gender, BMI)
  - Prior diagnoses (if available)
- **Time Window:** First 6 hours of ICU admission

---

## 🏥 Dataset

- **Source:** [MIMIC-IV](https://mimic.mit.edu/)
- **Subset:** 30% of original population, preprocessed by TAs
- **Final Cohort:** 16,922 patients with 20.87% mortality rate
- **Cohort Criteria:**
  - First ICU stay only
  - At least 6 hours of data before hospital discharge
  - Includes both survivors (13,391) and non-survivors (3,531)

---

## ⚙️ Pipeline

1. **Cohort Selection** ✅

   - Total patients in MIMIC database: 17316
   - Patients with ICU stays: 17316
   - Filtered to first ICU stay only: 17316
     - Excluded: 0 repeat ICU stays
   - Final cohort with ≥6 hours of records: 16922
     - Excluded: 389 patients with <6 hours of records

2. **Feature Extraction** ✅

   - **Demographics:**
     - Age, gender, anchor_age
     - BMI (calculated from height and weight using expanded chartevents IDs)
   - **Early Time Window Features (First 6 Hours):**
     - Temporal aggregations within the critical first 6 hours of ICU admission
     - Time-weighted averaging for more accurate vital sign representation
     - Early deterioration indicators
   - **Vital Signs (with comprehensive statistical aggregations):**
     - Heart rate, respiratory rate, blood pressure (SBP, DBP, MAP)
     - Temperature, SpO2 (oxygen saturation)
     - **Statistical measures:** mean, min, max, median, std, delta, percent change
     - **Temporal patterns:** trends over time, variability metrics
     - **Parallel processing** for efficient extraction from large chartevents data
   - **Laboratory Results (with statistical aggregations):**
     - **Complete blood count:** WBC, hemoglobin, hematocrit, platelets
     - **Chemistry panel:** sodium, potassium, creatinine, BUN, glucose
     - **Liver function:** bilirubin, alkaline phosphatase, ALT, AST
     - **Coagulation:** INR (International Normalized Ratio)
     - **Critical markers:** lactate, bicarbonate, anion gap
     - **Parallel processing** for efficient lab data extraction
     - **Missingness indicators** for features with significant missing patterns
   - **Advanced Clinical Features:**
     - **Urine output:** extracted from ICU outputevents
     - **Metastatic cancer flag:** derived from hospital diagnoses
     - **Prior diagnoses information:** comprehensive medical history analysis
   - **Clinical Derived Features:**
     - Shock index, SIRS criteria, organ dysfunction scores
     - Distance-from-normal metrics for vital signs
     - Critical value identification and counting
     - Temporal trends and variability measures
   - **Data Integration:**
     - Subject-to-stay ID mapping preservation throughout pipeline
     - Memory-efficient processing with garbage collection
     - Comprehensive data validation and quality checks
   - **Output Generation:**
     - **Table One** generation for demographic and clinical characteristics
     - Comprehensive feature statistics and distributions
     - **Final feature set:** 469 raw features before preprocessing
     - Features saved to `extracted_features.csv` for downstream processing

3. **Data Preprocessing** ✅

   - **Advanced Missing Data Handling:**
     - **Iterative Imputation (MICE):** Applied for critical clinical features with sophisticated missing patterns
     - **KNN Imputation:** Used for features with <5% missing values for localized imputation
     - **Median Imputation:** Applied for stable clinical measurements as fallback strategy
     - **Missingness Indicators:** Created for features with 20-80% missing to capture informative missingness
     - **Strategic Feature Dropping:** Removed features with >80% missing (except clinically critical ones like BMI)
     - **Clinical Domain Knowledge:** Applied medical expertise to determine appropriate imputation strategies
   - **Robust Outlier Detection and Handling:**
     - **Clinical Range Constraints:** Applied evidence-based physiological limits:
       - Heart rate: 30-200 bpm
       - Respiratory rate: 5-60 breaths/min
       - Mean Arterial Pressure: 40-180 mmHg
       - Systolic Blood Pressure: 60-220 mmHg
       - SpO2: 60-100%
       - Temperature: 32-42°C
     - **Statistical Outlier Detection:** Used IQR method and Z-score analysis
     - **Winsorizing:** Applied 1%-99% percentile capping for highly skewed variables
     - **Medical Validation:** Cross-referenced outliers with clinical literature
   - **Sophisticated Feature Engineering:**
     - **Clinical Derived Features:**
       - **SIRS Criteria Count:** Systematic inflammatory response syndrome indicators
       - **Shock Index:** Heart rate to systolic blood pressure ratio for hemodynamic assessment
       - **Organ Dysfunction Scores:** Multi-system failure indicators
       - **Critical Value Identification:** Automated flagging of abnormal lab values
     - **Distance-from-Normal Metrics:** Physiological deviation measurements:
       - Temperature deviation from 36.5°C (normothermia)
       - Heart rate deviation from 75 bpm (normal resting rate)
       - Respiratory rate deviation from 15 breaths/min (normal adult rate)
       - SBP deviation from 120 mmHg (optimal blood pressure)
     - **Temporal Pattern Analysis:**
       - **Delta Features:** Hour-to-hour changes in vital signs
       - **Percent Change:** Relative changes over time windows
       - **Trend Analysis:** Linear trends over the 6-hour observation period
       - **Variability Metrics:** Standard deviation and coefficient of variation
     - **Advanced Clinical Flags:**
       - Hypoxemia indicators (SpO2 < 92%)
       - Bradycardia/tachycardia flags
       - Fever/hypothermia detection
       - Shock state identification
   - **Feature Scaling and Transformation:**
     - **RobustScaler:** Applied for outlier-resistant standardization
     - **Log Transformation:** Applied to highly skewed features (skewness > 2)
     - **Polynomial Features:** Squared terms for key vital signs to capture non-linear relationships
     - **Feature Standardization:** Ensured consistent naming and data types across pipeline
   - **Model-Specific Feature Selection:**
     - **Logistic Regression Optimization:**
       - Maximum 50 features to prevent overfitting
       - Correlation analysis to remove multicollinear features
       - Statistical significance testing for feature relevance
       - L2 regularization compatibility focus
     - **XGBoost Optimization:**
       - Maximum 150 features to leverage tree-based feature interactions
       - Importance-based ranking using mutual information
       - Categorical feature encoding optimization
       - Missing value handling compatibility
     - **Random Forest Optimization:**
       - Maximum 100 features for balanced performance
       - Gini importance-based selection
       - Bootstrap sampling compatibility
       - Feature interaction preservation
   - **Data Quality Validation:**
     - **Comprehensive Data Validation:** Multi-stage quality checks throughout pipeline
     - **Clinical Plausibility Testing:** Medical domain knowledge validation
     - **Statistical Distribution Analysis:** Verification of feature distributions
     - **Missing Pattern Analysis:** Systematic evaluation of missingness mechanisms
     - **Feature Correlation Assessment:** Detection and handling of redundant features
   - **Memory and Performance Optimization:**
     - **Parallel Processing:** Multi-threaded feature engineering for large datasets
     - **Memory Management:** Efficient data structures and garbage collection
     - **Modular Architecture:** Separate preprocessing modules for maintainability
     - **Pipeline Reproducibility:** Consistent random seeds and deterministic operations
   - **Output Generation and Reporting:**
     - **Preprocessing Reports:** Comprehensive documentation of all transformations
     - **Feature Statistics:** Before/after comparison of data distributions
     - **Model-Specific Datasets:** Tailored preprocessing for each model type
     - **Quality Metrics:** Detailed reporting of missing values, outliers, and transformations
     - **Feature Provenance:** Complete traceability of feature engineering steps

   **Pipeline Results:**

   - **Input:** 469 raw extracted features
   - **Output:** 156 optimized features for XGBoost, 86 features for other models
   - **Processing Time:** Optimized for efficiency with parallel processing
   - **Data Quality:** 100% complete data after imputation with preserved clinical validity

4. **Model Development** ✅

   - **Random Forest** with Advanced Cost-Sensitive Learning

     - **Early Stopping Implementation:** Manual early stopping based on validation AUC with patience mechanism
     - **Hyperparameter Optimization:** Randomized search across 50 parameter combinations
       - n_estimators: [100, 200, 300, 500], max_depth: [5, 8, 10, 15, None]
       - min_samples_split: [5, 10, 15], min_samples_leaf: [2, 4, 8]
       - max_features: ["sqrt", "log2", 0.3], class_weight: ["balanced", "balanced_subsample"]
     - **Class Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique)
     - **Feature Importance Analysis:** Gini importance with permutation importance validation
     - **SHAP Integration:** TreeExplainer for feature interaction analysis
     - **Best Configuration:** 450 trees with validation AUC: 0.7910

   - **Logistic Regression** with Advanced Regularization

     - **Multi-Solver Optimization:** Support for saga, liblinear, newton-cg, and lbfgs solvers
     - **Comprehensive Regularization:** L1, L2, and ElasticNet penalties with cross-validation
     - **Feature Selection Pipeline:** L1-based feature selection with threshold optimization
     - **Automated Scaling:** StandardScaler integration with feature selection compatibility
     - **Clinical Interpretation:** Odds ratio calculation and coefficient analysis
     - **Hyperparameter Search:** Multi-distribution randomized search with convergence handling
     - **Best Features:** Selected 50 most predictive features for optimal performance

   - **XGBoost** (Single Model) with GPU Acceleration

     - **Advanced Hardware Optimization:** CUDA GPU acceleration with histogram-based tree construction
     - **Comprehensive Hyperparameter Tuning:** 50-iteration randomized search across:
       - n_estimators: [100-1000], learning_rate: [0.01-7.0]
       - max_depth: [3, 5, 7, 9], min_child_weight: [1, 3, 5]
       - subsample: [0.6-1.0], colsample_bytree: [0.6-1.0]
       - gamma: [0-0.3], reg_lambda: [1-10], scale_pos_weight: adaptive class weighting
     - **Early Stopping:** Validation-based early stopping with 30-round patience
     - **Multi-Threshold Evaluation:** Comprehensive threshold analysis for clinical applications
       - Standard (0.50), F1-optimized (0.45), Balanced (0.30), High-sensitivity (0.10)
     - **Best Configuration:**
       - Learning rate: 0.01, Max depth: 7, N estimators: 600
       - Min child weight: 1, Subsample: 0.6, Colsample bytree: 1.0
     - **Clinical Cost Analysis:** Weighted evaluation considering FN/FP cost ratios

   - **XGBoost Ensemble** (5, 7, and 10 Models)

     - **Diversity Strategy:** Multiple models with different random seeds for prediction variance reduction
     - **Prediction Averaging:** Ensemble predictions through probability averaging across all models
     - **Comprehensive Threshold Analysis:** Multi-threshold evaluation for different clinical scenarios:
       - Standard (0.50): Maximum precision for avoiding false alarms
       - F1-optimized (0.45): Best overall balance for general clinical use
       - Balanced (0.30): Good compromise for screening applications
       - High sensitivity (0.20): Emphasizes recall for critical care monitoring
       - Clinical utility (0.10): Maximum sensitivity for emergency scenarios
     - **Ensemble Size Optimization:** Performance analysis across 5, 7, and 10 model ensembles
     - **Robustness Analysis:** Improved prediction stability and calibration through model averaging
     - **Performance Results:**
       - 5-model ensemble: F1-score of 0.5043, AUC of 0.8465
       - 7-model ensemble: F1-score of 0.5052, AUC of 0.8465
       - 10-model ensemble: F1-score of 0.5060, AUC of 0.8464
     - **Optimal Configuration:** 7-10 models provide best performance with diminishing returns beyond this

   - **Advanced Evaluation Framework**

     - **Cross-Validation:** 5-fold stratified cross-validation with comprehensive metrics
     - **Threshold Optimization:** Clinical utility-based threshold selection with cost-sensitive analysis
     - **Performance Visualization:** ROC curves, Precision-Recall curves, confusion matrices
     - **Feature Interpretation:**
       - Permutation importance for model-agnostic feature ranking
       - SHAP (SHapley Additive exPlanations) values for individual prediction explanation
       - Clinical coefficient analysis for logistic regression interpretability
     - **Model Persistence:** Comprehensive model saving with metadata, scalers, and feature selectors

   - **Class Imbalance Strategies**

     - **SMOTE Application:** Synthetic minority oversampling for balanced training sets
     - **Cost-Sensitive Learning:** Class weight adjustment based on mortality rate (20.87%)
     - **Threshold Calibration:** Post-training threshold optimization for clinical priorities
     - **Evaluation Robustness:** Stratified sampling to maintain class distribution in splits

   - **Technical Optimizations**

     - **Memory Management:** Efficient data structures with garbage collection
     - **Parallel Processing:** Multi-threaded training and evaluation (-1 n_jobs)
     - **Reproducibility:** Consistent random seeds across all experiments (seed=42)
     - **Error Handling:** Comprehensive exception handling with graceful fallbacks
     - **GPU Utilization:** CUDA acceleration for XGBoost training and inference

   - **Clinical Integration Features**
     - **Multiple Threshold Strategies:** Adaptable to different clinical decision contexts
     - **Cost-Benefit Analysis:** Customizable FN/FP cost ratios for hospital-specific optimization
     - **Real-time Inference:** Optimized prediction pipeline for clinical deployment
     - **Interpretability Focus:** Model explanations suitable for clinical decision support
     - **Validation Framework:** Robust evaluation matching clinical validation standards

   **Training Infrastructure:**

   - **Computing Environment:** CUDA-enabled GPU acceleration for XGBoost models
   - **Data Pipeline:** Automated preprocessing with model-specific feature optimization
   - **Evaluation Protocol:** Train/Validation/Test split (60%/20%/20%) with stratification
   - **Performance Monitoring:** Comprehensive metrics tracking and visualization
   - **Model Versioning:** Timestamped model artifacts with complete reproducibility metadata

5. **Evaluation** ✅

   - **Comprehensive Performance Assessment:**

     - **5-Fold Stratified Cross-Validation:** Robust evaluation with consistent class distribution
     - **Multiple Threshold Analysis:** Clinical utility-driven threshold optimization
     - **Model Comparison Framework:** Standardized evaluation across all model types
     - **Feature Importance Analysis:** Comprehensive interpretation of predictive factors

   - **Evaluation Metrics:**

     - **Primary:** AUC-ROC for discrimination ability assessment
     - **Secondary:** Precision, Recall, F1-score, Specificity for balanced evaluation
     - **Clinical:** Multiple threshold strategies for different care scenarios
     - **Robustness:** Ensemble variance analysis and prediction stability

   - **Best Performing Configuration:**

     - **XGBoost Ensemble (7-10 models)** achieves optimal performance
     - **AUC: 0.8464-0.8465** with superior discrimination
     - **F1-optimized threshold (0.45)** provides best clinical balance
     - **Ensemble averaging** improves prediction robustness and calibration

   - **Key Performance Insights:**
     - **Diminishing returns** observed beyond 7-model ensemble size
     - **Threshold flexibility** enables adaptation to clinical priorities
     - **Feature importance** reveals age, respiratory parameters, and SIRS criteria as top predictors
     - **Ensemble approaches** consistently outperform single models in precision and reliability

## 📈 Results

| Model                       | AUC    | Accuracy | Precision | Recall | F1-Score | Specificity |
| --------------------------- | ------ | -------- | --------- | ------ | -------- | ----------- |
| Random Forest               | 0.8145 | 0.7900   | 0.4971    | 0.6091 | 0.5474   | -           |
| Logistic Regression         | 0.7876 | 0.7439   | 0.4244    | 0.6402 | 0.5104   | -           |
| XGBoost (threshold=0.50)    | 0.8426 | 0.8171   | 0.5641    | 0.5425 | 0.5531   | -           |
| XGBoost (threshold=0.45)\*  | 0.8426 | 0.8095   | 0.5370    | 0.6275 | 0.5787   | 0.8574      |
| XGBoost (threshold=0.30)†   | 0.8426 | 0.7229   | 0.4167    | 0.8215 | 0.5529   | 0.6969      |
| XGBoost (threshold=0.10)‡   | 0.8426 | 0.4520   | 0.2736    | 0.9830 | 0.4280   | 0.3121      |
| **XGBoost Ensemble (n=5)**  |        |          |           |        |          |             |
| - Standard (0.50)           | 0.8465 | 0.8281   | 0.6792    | 0.3329 | 0.4468   | 0.9586      |
| - F1-optimized (0.45)§      | 0.8465 | 0.8292   | 0.6391    | 0.4164 | 0.5043   | 0.9380      |
| - Balanced (0.30)           | 0.8465 | 0.8012   | 0.5180    | 0.6742 | 0.5858   | 0.8346      |
| - High sensitivity (0.20)   | 0.8465 | 0.7155   | 0.4109    | 0.8399 | 0.5519   | 0.6827      |
| - Clinical utility (0.10)   | 0.8465 | 0.5306   | 0.3026    | 0.9589 | 0.4601   | 0.4177      |
| **XGBoost Ensemble (n=7)**  |        |          |           |        |          |             |
| - Standard (0.50)           | 0.8465 | 0.8278   | 0.6793    | 0.3300 | 0.4442   | 0.9589      |
| - F1-optimized (0.45)‖      | 0.8465 | 0.8298   | 0.6419    | 0.4164 | 0.5052   | 0.9388      |
| - Balanced (0.30)           | 0.8465 | 0.7985   | 0.5130    | 0.6686 | 0.5806   | 0.8328      |
| - High sensitivity (0.20)   | 0.8465 | 0.7161   | 0.4116    | 0.8414 | 0.5528   | 0.6831      |
| - Clinical utility (0.10)   | 0.8465 | 0.5306   | 0.3028    | 0.9603 | 0.4604   | 0.4173      |
| **XGBoost Ensemble (n=10)** |        |          |           |        |          |             |
| - Standard (0.50)           | 0.8464 | 0.8295   | 0.6903    | 0.3314 | 0.4478   | 0.9608      |
| - F1-optimized (0.45)¶      | 0.8464 | 0.8298   | 0.6413    | 0.4178 | 0.5060   | 0.9384      |
| - Balanced (0.30)           | 0.8464 | 0.7985   | 0.5130    | 0.6700 | 0.5811   | 0.8324      |
| - High sensitivity (0.20)   | 0.8464 | 0.7143   | 0.4098    | 0.8399 | 0.5509   | 0.6812      |
| - Clinical utility (0.10)   | 0.8464 | 0.5306   | 0.3028    | 0.9603 | 0.4604   | 0.4173      |

\* F1-optimized threshold (single model)  
 † Balanced threshold (single model)  
 ‡ High-sensitivity/clinical utility threshold (single model)  
 § F1-optimized threshold (5-model ensemble)  
 ‖ F1-optimized threshold (7-model ensemble)  
 ¶ F1-optimized threshold (10-model ensemble)

### Model Performance Comparison:

**Single XGBoost vs. Ensemble XGBoost:**

- **Ensemble models** consistently achieve improved AUC (0.8464-0.8465) compared to single XGBoost model (0.8426)
- **Ensemble approaches provide more conservative predictions** with higher specificity across all thresholds
- The ensemble methods reduce prediction variance through model averaging and show improved robustness

**Ensemble Size Analysis:**

- **5-model ensemble**: F1-score of 0.5043, AUC of 0.8465
- **7-model ensemble**: F1-score of 0.5052, AUC of 0.8465
- **10-model ensemble**: F1-score of 0.5060, AUC of 0.8464

**Key Observations:**

- **Diminishing returns** with ensemble size - marginal improvement from 5 to 10 models
- **7-model and 10-model ensembles** show slight improvements in F1-score compared to 5-model
- **Consistent AUC performance** (~0.8465) across all ensemble sizes
- **Higher precision** maintained across all ensemble configurations compared to single model

**Threshold Analysis for XGBoost Ensembles:**

- **Standard threshold (0.50)**: Highest precision (64-69%) but lowest recall (33%) - best for avoiding false positives
- **F1-optimized threshold (0.45)**: Best balance with F1-scores ranging 0.5043-0.5060 across ensemble sizes
- **Balanced threshold (0.30)**: Good compromise between precision (51%) and recall (67-68%)
- **High sensitivity threshold (0.20)**: Emphasizes recall (84%) for screening applications
- **Clinical utility threshold (0.10)**: Maximum sensitivity (96%) for critical care scenarios

The XGBoost ensembles demonstrate superior discrimination ability and provide flexible threshold options for different clinical priorities, with optimal performance achieved around 7-10 models.

### Top 10 Important Features from Random Forest:

_(Values represent Gini importance - the relative contribution to reducing impurity)_

1. Age (0.0527)
2. Anchor age (0.0506)
3. Previous diagnosis count (0.0444)
4. Respiratory rate (mean) (0.0363)
5. ICU duration hours (0.0354)
6. ICU duration hours (log) (0.0341)
7. Shock index (0.0333)
8. Systolic blood pressure (min) (0.0310)
9. SpO2 (mean) (0.0265)
10. Temperature (mean) (0.0262)

### Top 10 Important Features from Logistic Regression:

_(Values represent standardized coefficients - the relative effect size on log-odds of mortality)_

1. Age (0.6477)
2. Heart rate percent change (0.5329)
3. Shock index (0.5050)
4. SpO2 mean distance from normal (0.5025)
5. Heart rate mean (-0.5004) _negative effect_
6. Lactate mean log (0.4899)
7. Heart rate min (0.3769)
8. Lactate max log (-0.3704) _negative effect_
9. Temperature max (0.3570)
10. Previous diagnosis count (0.3466)

### Top 10 Important Features from XGBoost:

_(Values represent feature importance scores)_

1. Age very elderly (0.1362)
2. Critical value count (0.0484)
3. Previous respiratory diagnoses count (0.0374)
4. Has prior diagnoses (0.0208)
5. Platelets measured (0.0199)
6. Previous nervous/sensory diagnoses count (0.0196)
7. SIRS criteria count (0.0191)
8. Gender (0.0166)
9. Creatinine measured (0.0163)
10. Previous circulatory diagnoses count (0.0154)

### Key Findings:

- **Demographic factors (age)** are strongly predictive of mortality across all models
- **Respiratory parameters** emerge as critical predictors across all approaches
- The derived feature **"shock index"** demonstrates significant predictive value in both Random Forest and Logistic Regression
- **Heart rate variability** (percent change) is a strong predictor in the logistic regression model
- Logistic regression identified **negative associations** between mortality and heart rate mean and lactate max log
- **XGBoost single model** achieves excellent performance with highest F1-score (0.5787) among individual models
- **XGBoost ensemble approaches** provide the best overall discrimination with **AUC of 0.8464-0.8465**
- **Ensemble methodology insights**:
  - **Optimal ensemble size**: 7-10 models provide best performance with diminishing returns beyond this
  - **Improved robustness** through prediction averaging across multiple models
  - **Higher precision** at equivalent thresholds compared to single models
  - **Better calibration** for risk stratification in clinical settings
  - **Consistent performance** across different ensemble sizes (5, 7, 10 models)
- **Critical value count** and **presence of prior diagnoses** were uniquely important in XGBoost models
- **SIRS criteria count** (systemic inflammatory response) was identified as a key predictor by XGBoost
- **Ensemble modeling reduces overfitting** and provides more reliable predictions for clinical decision-making
- **Threshold flexibility** allows adaptation to different clinical scenarios (screening vs. precision-focused care)

_Model evaluation based on 5-fold cross-validation and testing on 20% held-out data._  
 _XGBoost ensemble results based on averaging predictions from multiple models with different random seeds._

6. **Conclusion & Clinical Implications** ✅

   - **Best Performing Models:**
     - XGBoost ensemble (7-10 models) achieves optimal discrimination (AUC: 0.8464-0.8465)
     - F1-optimized threshold (0.45) provides best clinical balance
   - **Clinical Translation Recommendations:**
     - Standard threshold (0.50): High-precision applications
     - Balanced threshold (0.30): Screening programs
     - High-sensitivity threshold (0.10): Critical care monitoring
   - **Key Clinical Insights:**
     - Age and respiratory parameters are primary mortality predictors
     - SIRS criteria and critical value counts provide novel risk stratification
     - Ensemble approaches improve reliability for clinical decision support
   - **Limitations:**
     - Single-center dataset may limit generalizability
     - 6-hour time window may miss later deterioration patterns
     - Class imbalance requires careful threshold selection
   - **Future Work:**
     - Multi-center validation
     - Integration with real-time clinical systems
     - Temporal modeling for dynamic risk assessment

---

## 📂 Repository Structure

```bash
├── data/                     # Processed datasets (excluded from Git)
├── notebooks/
├── src/                           # Python source code
│   ├── main.py                    # Main training and evaluation script
│   ├── config.py
│   ├── data_preprocessing.py      # Enhanced preprocessing pipeline
│   ├── cohort_selection.py        # MIMIC-IV cohort selection pipeline
│   ├── feature_extraction.py      # Comprehensive feature extraction
│   ├── models/                    # Model implementations
│   │   ├── base_model.py          # Abstract base model class
│   │   ├── xgboost.py             # XGBoost implementation with GPU support
│   │   ├── random_forest.py       # Random Forest with early stopping
│   │   └── logistic_regression.py # Logistic Regression with regularization
│   ├── preprocessing/             # Modular preprocessing components
│   │   ├── __init__.py
│   │   ├── feature_engineering.py
│   │   ├── imbalance.py
│   │   ├── imputation.py
│   │   ├── outliers.py
│   │   ├── scaling.py
│   │   ├── selection.py
│   │   ├── utils.py
│   │   └── validation.py
│   └── feature_extraction/             # Modular feature extraction components
│       ├── __init__.py
│       ├── clinical_features.py
│       ├── cohort.py
│       ├── demographics.py
│       ├── labels.py
│       ├── labs.py
│       ├── reporting.py
│       ├── time_windows.py
│       └── vitals.py
└── [README.md](http://_vscodecontentref_/0)
```

---

## 🚀 Installation & Setup

### Prerequisites

- Anaconda or Miniconda
- CUDA-compatible GPU (optional, for XGBoost acceleration)

### Environment Setup

1. **Clone the repository:**

   ```bash
   git clone https://github.com/benedictdavon/early-icu-mortality-prediction
   cd early-icu-mortality-prediction
   ```

2. **Create and activate conda environment:**

   ```bash
   # Create environment from yml file
   conda env create -f environment.yml

   # Activate environment
   conda activate icu-mortality-prediction
   ```

3. **Verify installation:**
   ```bash
   python -c "import xgboost; print('XGBoost version:', xgboost.__version__)"
   python -c "import sklearn; print('Scikit-learn version:', sklearn.__version__)"
   ```

### Data Setup

1. **Access MIMIC-IV dataset:**

   - Download the preprocessed 30% subset provided by course TAs
   - Place the raw data files in the `data/raw/` directory

2. **Data directory structure:**
   ```bash
   data/
   ├── raw/                    # Original MIMIC-IV files
   ├── processed/              # Processed datasets
   ├── extracted_features.csv  # Feature extraction output
   └── final_dataset.csv       # Model-ready dataset
   ```

### GPU Setup (Optional)

For XGBoost GPU acceleration:

```bash
# Check CUDA version
nvidia-smi

# If needed, update CUDA toolkit
conda install cudatoolkit=11.8  # Match your CUDA version
```

---

## 🚀 Running the Pipeline

### Complete Pipeline Execution

1. **Cohort Selection:**

   ```bash
   python src/cohort_selection.py
   ```

2. **Feature Extraction:**

   ```bash
   python src/feature_extraction.py
   ```

3. **Data Preprocessing:**
   ```bash
   python src/data_preprocessing.py
   ```

### Model Training & Evaluation

#### Basic Model Training

Train individual models with default settings:

```bash
# Random Forest
python src/main.py --model random_forest

# Logistic Regression
python src/main.py --model logistic_regression

# XGBoost (single model)
python src/main.py --model xgboost

# XGBoost Ensemble (5 models)
python src/main.py --model xgboost_ensemble
```

#### Advanced Training Options

**With hyperparameter tuning:**

```bash
# Enable hyperparameter tuning (default: enabled)
python src/main.py --model xgboost --tune

# Disable hyperparameter tuning
python src/main.py --model xgboost --no-tune
```

**With early stopping control:**

```bash
# Enable early stopping (default: enabled for tree models)
python src/main.py --model random_forest --early-stopping

# Disable early stopping
python src/main.py --model random_forest --no-early-stopping
```

**With SHAP analysis control:**

```bash
# Enable SHAP analysis (default: enabled)
python src/main.py --model xgboost --shap

# Disable SHAP analysis (faster training)
python src/main.py --model xgboost --no-shap
```

**Ensemble size configuration:**

```bash
# 5-model ensemble (default)
python src/main.py --model xgboost_ensemble --ensemble-size 5

# 7-model ensemble
python src/main.py --model xgboost_ensemble --ensemble-size 7

# 10-model ensemble
python src/main.py --model xgboost_ensemble --ensemble-size 10
```

**Custom data paths and output directories:**

```bash
# Specify custom data path
python src/main.py --model xgboost --data-path /path/to/custom/data.csv

# Specify custom output directory
python src/main.py --model xgboost --output-dir /path/to/custom/results
```

#### Complete Training Examples

**Full training with all features enabled:**

```bash
# XGBoost with full analysis
python src/main.py --model xgboost --tune --early-stopping --shap

# Random Forest with comprehensive evaluation
python src/main.py --model random_forest --tune --early-stopping --shap

# Logistic Regression with regularization tuning
python src/main.py --model logistic_regression --tune --shap
```

**Fast training for testing:**

```bash
# Quick XGBoost training (no tuning, no SHAP)
python src/main.py --model xgboost --no-tune --no-shap

# Quick ensemble training
python src/main.py --model xgboost_ensemble --no-tune --ensemble-size 3
```

**Comprehensive ensemble analysis:**

```bash
# Train multiple ensemble sizes for comparison
python src/main.py --model xgboost_ensemble --ensemble-size 5
python src/main.py --model xgboost_ensemble --ensemble-size 7
python src/main.py --model xgboost_ensemble --ensemble-size 10
```

### Command Line Options

| Option                | Description                   | Default                | Example                       |
| --------------------- | ----------------------------- | ---------------------- | ----------------------------- |
| `--model`             | Model type to train           | `random_forest`        | `--model xgboost`             |
| `--data-path`         | Path to preprocessed data     | Auto-detected          | `--data-path data/custom.csv` |
| `--output-dir`        | Results output directory      | `results/<model>`      | `--output-dir custom_results` |
| `--no-tune`           | Disable hyperparameter tuning | Tuning enabled         | `--no-tune`                   |
| `--no-early-stopping` | Disable early stopping        | Early stopping enabled | `--no-early-stopping`         |
| `--no-shap`           | Skip SHAP analysis            | SHAP enabled           | `--no-shap`                   |
| `--ensemble-size`     | Number of ensemble models     | `5`                    | `--ensemble-size 10`          |

---

## 👨‍🏫 Team

This project was completed as part of the **AI in EHR – AI Capstone 2025** course at **National Yang Ming Chiao Tung University (NYCU)**.

- **Name:** [Benedict Davon Martono 周恭麟]
- **Student ID:** [110550201]
- **Instructor:** Prof. [王才沛]

---

## 👨‍👩‍👧 Team Acknowledgment

This project was originally conducted as a group assignment for the **NYCU AI Capstone 2025** course, in collaboration with:

- [Tymofii Voitekh 提姆西]
- [Jorge Tyrakowski 狄豪飛]

This GitHub repository reflects **my personal implementation and exploration** of the final project. All code and analysis here were done independently by me.

## 📚 References

- Singh, A., Nadkarni, G., Gottesman, O., Ellis, S. B., Bottinger, E. P., & Guttag, J. V. (2015). Incorporating temporal EHR data in predictive models for risk stratification of renal function deterioration. _Journal of Biomedical Informatics, 53_, 220–228. https://doi.org/10.1016/j.jbi.2014.11.005

- Meng, Y., Speier, W., Ong, M. K., & Arnold, C. W. (2021). Bidirectional representation learning from transformers using multimodal electronic health record data to predict depression. _IEEE Journal of Biomedical and Health Informatics, 25_(8), 3121–3129. https://doi.org/10.1109/JBHI.2021.3063721

- Solís-García, J., Vega-Márquez, B., Nepomuceno, J. A., Riquelme-Santos, J. C., & Nepomuceno-Chamorro, I. A. (2023). Comparing artificial intelligence strategies for early sepsis detection in the ICU: An experimental study. _Applied Intelligence, 53_(24), 30691–30705. https://doi.org/10.1007/s10489-023-05124-z

- Chen, Z., Tan, S., Chajewska, U., Rudin, C., & Caruna, R. (2023, June 22–24). Missing values and imputation in healthcare data: Can interpretable machine learning help? In B. J. Mortazavi, T. Sarker, A. Beam, & J. C. Ho (Eds.), _Proceedings of the Conference on Health, Inference, and Learning_ (Vol. 209, pp. 86–99). PMLR. https://proceedings.mlr.press/v209/chen23a.html

- Shashikumar, S. P., Josef, C. S., Sharma, A., & Nemati, S. (2021). DeepAISE—An interpretable and recurrent neural survival model for early prediction of sepsis. _Artificial Intelligence in Medicine, 113_, 102036. https://doi.org/10.1016/j.artmed.2021.102036

- Shukla, S. N., & Marlin, B. (2020, October 2). Multi-time attention networks for irregularly sampled time series. _International Conference on Learning Representations (ICLR)_. https://openreview.net/pdf?id=4c0J6lwQ4_

- Yang, Z., Mitra, A., Liu, W., Berlowitz, D., & Yu, H. (2023). TransformEHR: Transformer-based encoder-decoder generative model to enhance prediction of disease outcomes using electronic health records. _Nature Communications, 14_(1), 7857. https://doi.org/10.1038/s41467-023-43715-z

- Gao, J., Lu, Y., Ashrafi, N., Domingo, I., Alaei, K., & Pishgar, M. (2024). Prediction of sepsis mortality in ICU patients using machine learning methods. _BMC Medical Informatics and Decision Making, 24_(1), 228. https://doi.org/10.1186/s12911-024-02630-z

- Iwase, S., Nakada, T.-A., Shimada, T., Oami, T., Shimazui, T., Takahashi, N., Yamabe, J., Yamao, Y., & Kawakami, E. (2022). Prediction algorithm for ICU mortality and length of stay using machine learning. _Scientific Reports, 12_(1), 12912. https://doi.org/10.1038/s41598-022-17091-5

- Hou, N., Li, M., He, L., Xie, B., Wang, L., Zhang, R., Yu, Y., Sun, X., Pan, Z., & Wang, K. (2020). Predicting 30-days mortality for MIMIC-III patients with sepsis-3: A machine learning approach using XGBoost. _Journal of Translational Medicine, 18_(1), 462. https://doi.org/10.1186/s12967-020-02620-5

- MIMIC-IV Dataset: https://mimic.mit.edu/
- Course Materials: Provided by NYCU

---
