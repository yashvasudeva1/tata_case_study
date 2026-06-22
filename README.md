# Credit Delinquency Prediction: A Statistical and Predictive Analysis

## Project Overview
This repository contains a comprehensive exploratory data analysis (EDA), statistical hypothesis testing, and machine learning benchmarking pipeline for a customer delinquency dataset. 

The primary objective of this project was to determine if customer demographic, financial, and behavioral features could reliably predict the `Delinquent_Account` target variable. Through rigorous statistical testing and cross-validated model evaluation, the analysis concludes that the dataset lacks predictive signal, serving as a documented case study on identifying randomly or independently assigned target labels in tabular data.

## Features Analyzed
The dataset contains a mix of numerical and categorical variables across three domains:
* **Demographics & Status:** `Age`, `Employment_Status`, `Location`, `Account_Tenure`
* **Financial Metrics:** `Income`, `Credit_Score`, `Credit_Utilization`, `Loan_Balance`, `Debt_to_Income_Ratio`, `Credit_Card_Type`
* **Behavioral/Historical:** `Missed_Payments`, `Month_1` to `Month_6` payment statuses.

## Methodology & Findings

### 1. Statistical Hypothesis Testing
To isolate feature importance before modeling, two statistical tests were conducted with a significance threshold of **α = 0.05**:

* **Welch's t-test (Numeric Features):** Evaluated whether the means of numerical features differed significantly between delinquent and non-delinquent accounts. 
* **Chi-square Test (Categorical Features):** Evaluated the independence of categorical features against the target.

**Result:** **0 out of 17** features showed statistical significance.
* *Highest performing numeric feature:* `Missed_Payments` (p = 0.5621)
* *Highest performing categorical feature:* `Month_1` (p = 0.1058)

### 2. Correlation Analysis
Pearson correlation coefficients (r) were calculated to detect linear relationships between numerical features and the target.
* The correlations were exceptionally weak across the board, ranging from `-0.040` (`Account_Tenure`) to `0.045` (`Income`).
* Notably, features that logically should correlate strongly with delinquency (e.g., `Credit_Score` at `0.035`, `Missed_Payments` at `-0.026`) showed virtually zero linear relationship.

### 3. Machine Learning Benchmarks
Several classification models were trained and evaluated using 5-fold Cross Validation to account for class imbalance and test generalizability. 

| Model | Mean AUC | AUC Std Dev | Mean Accuracy | Mean F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.471 | 0.056 | 0.514 | 0.211 |
| **Random Forest** | 0.384 | 0.050 | 0.840 | 0.000 |
| **Gradient Boosting** | 0.374 | 0.088 | 0.806 | 0.040 |
| **Baseline (Majority Class)** | **0.500** | **0.000** | **0.840** | **0.000** |

*Note: The dataset exhibits a strict 84/16 class imbalance. The models fail to outperform a naive baseline model that simply predicts the majority class ("Not Delinquent") 100% of the time.*

## Final Verdict
The empirical evidence from this analysis points to a conclusive **null result**:

1. **No feature shows a statistically significant relationship** with the target variable.
2. **No machine learning model exceeds random chance** (AUC 0.50) on cross-validation. 
3. The achievable performance ceiling for any predictive architecture trained on this specific data matrix is firmly capped at the majority-class baseline.

**Conclusion:** The `Delinquent_Account` target does not appear to be mathematically derived from the provided feature set. The label is likely a randomly or independently assigned artifact. This repository serves as a robust framework for proving the absence of signal in synthetic or detached datasets.

---
*Maintained by Yash Vasudeva*
