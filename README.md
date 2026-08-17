# 2025AC05271 ML Assignment 2

## a. Problem Statement

A retail bank runs telephone campaigns to promote term-deposit products, yet only a small fraction of contacted customers actually subscribe. Phoning every lead is expensive and inefficient, so the bank needs a way to rank prospects before the campaign begins. This project tackles that challenge as a supervised binary-classification task: given client demographics, financial attributes, and campaign-history variables, predict whether a contact will end in a subscription (`yes`) or not (`no`). Five fundamentally different learning algorithms are fitted on identical train/test splits and benchmarked side by side with six complementary metrics (Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient), so the final recommendation rests on evidence rather than intuition.

---

## b. Dataset Description

| Property | Details |
|----------|---------|
| **Name** | Bank Marketing Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Instances** | 45,211 client-contact records |
| **Input Features** | 16 (9 categorical + 7 numeric) |
| **Target Variable** | `y` — term deposit subscription (`yes` = 1, `no` = 0) |
| **Class Balance** | Skewed, roughly 11.5% subscribers vs 88.5% non-subscribers |
| **Notable Columns** | `duration` (seconds of the final call), `campaign` (contacts made this campaign), `pdays` (days since previous contact), `balance` (average yearly balance in euros), plus `job`, `marital`, `education`, `housing`, `loan` |
| **Use Case** | Prioritizing call lists for a Portuguese bank's direct-marketing operation |

---

## c. GitHub Repository Link

https://github.com/bikram-banerjee/AIMLCZG565_MLAssignment2.git

> **Repository contents:** `app.py`, `requirements.txt`, `README.md`, `test_data.csv`, and the `model/` directory containing training scripts, inference utilities, serialized `.joblib` pipelines, and `metrics.json`.

---

## d. Models Used

### Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|-----|-----------|--------|----|-----|
| **Logistic Regression** | 0.8457 | 0.9079 | 0.4182 | 0.8147 | 0.5527 | 0.5092 |
| **Decision Tree** | 0.8782 | 0.6933 | 0.4785 | 0.4518 | 0.4648 | 0.3963 |
| **kNN** | 0.8962 | 0.8277 | 0.599 | 0.3403 | 0.434 | 0.4001 |
| **Naive Bayes** | 0.8548 | 0.8101 | 0.4059 | 0.5198 | 0.4559 | 0.3774 |
| **Random Forest (Ensemble)** | 0.9051 | 0.9286 | 0.6938 | 0.3384 | 0.4549 | 0.4415 |

---

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Delivers a transparent linear benchmark and, thanks to balanced class weights, posts the highest recall of the group (0.81). Its straight-line decision surface, however, cannot bend around interactions such as `duration` × `campaign`, so precision and AUC stay in the middle of the pack. |
| **Decision Tree** | Produces rules a marketing manager could read aloud, but the unconstrained splits latch onto noise in high-cardinality fields like `job`. The result is the weakest AUC of the five (0.6933) and visible overfitting relative to the ensemble. |
| **kNN** | Once the nine categorical inputs are one-hot encoded, each record becomes a point in a very sparse, high-dimensional space where Euclidean distance loses meaning. The classifier still reaches 0.8962 accuracy by leaning on the majority class, yet its recall of 0.3403 shows it misses most true subscribers. |
| **Naive Bayes** | Trains in a fraction of a second, but the independence assumption behind it clashes with correlated banking attributes (`housing`, `loan`) and with mutually exclusive one-hot columns. Its probability outputs are therefore poorly calibrated — acceptable recall, bottom-tier precision. |
| **Random Forest (Ensemble)** | Bagging hundreds of decorrelated trees smooths out the variance that hurt the single tree and handles the mixed numeric/categorical space natively. It tops the table on AUC (0.9286) and Accuracy (0.9051) and achieves the best MCC, indicating the most balanced behavior on this skewed target. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)**. It leads on AUC, Accuracy, and MCC simultaneously, which makes it the safest choice for ranking campaign leads in production. |

---

## Additional Files Summary

| File / Folder | Purpose |
|---------------|---------|
| `app.py` | Streamlit interactive web application for single and batch predictions, with confusion matrix and classification report on the test set |
| `requirements.txt` | Python dependencies (scikit-learn, pandas, streamlit, joblib, matplotlib, etc.) |
| `test_data.csv` | Stratified hold-out test sample (auto-generated during training) |
| `model/train_and_save.py` | End-to-end training script that downloads data, fits all 5 pipelines, evaluates metrics, and saves `.joblib` models |
| `model/predict.py` | Standalone CLI utility for batch inference |
| `model/metrics.json` | JSON file containing all computed evaluation metrics |
| `model/*.joblib` | Serialized preprocessing + classifier pipelines for deployment |

---

*This README content is also included in the submitted PDF report.*
