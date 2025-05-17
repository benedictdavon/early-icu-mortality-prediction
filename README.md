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
- **Cohort Criteria:**
  - First ICU stay only
  - At least 6 hours of data before hospital discharge
  - Includes both survivors and non-survivors

---

## ⚙️ Pipeline

1. **Cohort Selection**
2. **Feature Extraction**
3. **Data Preprocessing**
   - Handling missing values and outliers
   - Normalization
   - Time-window aggregation
4. **Model Development**
   - Classical models (e.g., Logistic Regression, XGBoost)
   - Deep learning (optional)
5. **Evaluation**
   - Metrics: Accuracy, Precision, Recall, F1-score, AUC
   - Cross-validation and test split
6. **Conclusion & Insights**

---

## 📈 Results

| Model               | AUC   | Accuracy | F1-Score |
|--------------------|-------|----------|----------|
| Logistic Regression| 0.xxx | 0.xxx    | 0.xxx    |
| XGBoost            | 0.xxx | 0.xxx    | 0.xxx    |

_Results are preliminary and subject to tuning._

---

## 📂 Repository Structure

```bash
├── data/                 # Processed datasets (excluded from Git)
├── notebooks/            # Jupyter notebooks for EDA, preprocessing, training
├── src/                  # Python scripts for modeling and preprocessing
├── results/              # Evaluation results, plots, and outputs
├── figures/              # Diagrams and flowcharts
├── README.md
```

---

## 📊 Visualization

- Cohort selection flowchart  
- Feature distributions  
- Model performance ROC curves  

_(See `/figures` or `/notebooks/EDA.ipynb`)_

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

- Singh, A., Nadkarni, G., Gottesman, O., Ellis, S. B., Bottinger, E. P., & Guttag, J. V. (2015). Incorporating temporal EHR data in predictive models for risk stratification of renal function deterioration. *Journal of Biomedical Informatics, 53*, 220–228. https://doi.org/10.1016/j.jbi.2014.11.005

- Meng, Y., Speier, W., Ong, M. K., & Arnold, C. W. (2021). Bidirectional representation learning from transformers using multimodal electronic health record data to predict depression. *IEEE Journal of Biomedical and Health Informatics, 25*(8), 3121–3129. https://doi.org/10.1109/JBHI.2021.3063721

- Solís-García, J., Vega-Márquez, B., Nepomuceno, J. A., Riquelme-Santos, J. C., & Nepomuceno-Chamorro, I. A. (2023). Comparing artificial intelligence strategies for early sepsis detection in the ICU: An experimental study. *Applied Intelligence, 53*(24), 30691–30705. https://doi.org/10.1007/s10489-023-05124-z

- Chen, Z., Tan, S., Chajewska, U., Rudin, C., & Caruna, R. (2023, June 22–24). Missing values and imputation in healthcare data: Can interpretable machine learning help? In B. J. Mortazavi, T. Sarker, A. Beam, & J. C. Ho (Eds.), *Proceedings of the Conference on Health, Inference, and Learning* (Vol. 209, pp. 86–99). PMLR. https://proceedings.mlr.press/v209/chen23a.html

- Shashikumar, S. P., Josef, C. S., Sharma, A., & Nemati, S. (2021). DeepAISE—An interpretable and recurrent neural survival model for early prediction of sepsis. *Artificial Intelligence in Medicine, 113*, 102036. https://doi.org/10.1016/j.artmed.2021.102036

- Shukla, S. N., & Marlin, B. (2020, October 2). Multi-time attention networks for irregularly sampled time series. *International Conference on Learning Representations (ICLR)*. https://openreview.net/pdf?id=4c0J6lwQ4_

- Yang, Z., Mitra, A., Liu, W., Berlowitz, D., & Yu, H. (2023). TransformEHR: Transformer-based encoder-decoder generative model to enhance prediction of disease outcomes using electronic health records. *Nature Communications, 14*(1), 7857. https://doi.org/10.1038/s41467-023-43715-z

- Gao, J., Lu, Y., Ashrafi, N., Domingo, I., Alaei, K., & Pishgar, M. (2024). Prediction of sepsis mortality in ICU patients using machine learning methods. *BMC Medical Informatics and Decision Making, 24*(1), 228. https://doi.org/10.1186/s12911-024-02630-z

- Iwase, S., Nakada, T.-A., Shimada, T., Oami, T., Shimazui, T., Takahashi, N., Yamabe, J., Yamao, Y., & Kawakami, E. (2022). Prediction algorithm for ICU mortality and length of stay using machine learning. *Scientific Reports, 12*(1), 12912. https://doi.org/10.1038/s41598-022-17091-5

- Hou, N., Li, M., He, L., Xie, B., Wang, L., Zhang, R., Yu, Y., Sun, X., Pan, Z., & Wang, K. (2020). Predicting 30-days mortality for MIMIC-III patients with sepsis-3: A machine learning approach using XGBoost. *Journal of Translational Medicine, 18*(1), 462. https://doi.org/10.1186/s12967-020-02620-5

- MIMIC-IV Dataset: https://mimic.mit.edu/
- Course Materials: Provided by NYCU

---
