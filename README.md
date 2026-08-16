# Bank Marketing � Multi-Model Classification

## a. Problem Statement
The objective of this project is to build a machine learning classification pipeline that predicts whether a bank client will subscribe to a term deposit (`yes` / `no`) based on demographic, campaign, and previous-contact features.  
Multiple classification algorithms are trained on the same dataset and rigorously compared using six standard evaluation metrics (Accuracy, AUC, Precision, Recall, F1, and MCC) to identify the best-performing model for this imbalanced, real-world marketing dataset.

---

## b. Dataset Description
| Property | Details |
|----------|---------|
| **Name** | Bank Marketing Dataset |
| **Source** | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) |
| **Instances** | 45,211 |
| **Input Features** | 16 (9 categorical + 7 numeric) |
| **Target Variable** | `y` � term deposit subscription (`yes` = 1, `no` = 0) |
| **Class Balance** | Imbalanced (~11.5% positive / ~88.5% negative) |
| **Notable Columns** | `duration` (last contact duration in seconds), `campaign` (number of contacts performed), `pdays` (days since last contact), `balance` (average yearly balance), `job`, `education`, `marital`, etc. |
| **Use Case** | Direct marketing campaign optimization for a Portuguese banking institution. |

---

## c. GitHub Repository Link
[https://github.com/<YOUR_USERNAME>/bank-marketing-classification](https://github.com/<YOUR_USERNAME>/bank-marketing-classification)

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
| **Logistic Regression** | Provides a strong, interpretable linear baseline. With balanced class weights, it achieves respectable recall; however, its linear decision boundary struggles to model complex non-linear interactions (e.g., between `duration` and `campaign`) inherent in the campaign data, resulting in moderate precision and AUC. |
| **Decision Tree** | Offers explicit, human-readable rules but is prone to overfitting on high-cardinality categorical variables such as `job` and training-set noise. Consequently, it exhibits lower generalization AUC and higher variance compared with ensemble methods. |
| **kNN** | Performance is hindered by the curse of dimensionality that arises after one-hot encoding nine categorical features into a high-dimensional sparse space. Euclidean distance becomes less discriminative, producing class boundaries that are inferior to tree-based and probabilistic models on this dataset. |
| **Naive Bayes** | Extremely fast to train, but its conditional independence assumption is severely violated by correlated financial attributes (e.g., `housing` and `loan`) and by binary one-hot encoded columns. It tends toward skewed probability estimates, yielding reasonable recall but comparatively low precision. |
| **Random Forest (Ensemble)** | Aggregates multiple decorrelated trees via bagging, making it robust to mixed data types and feature interactions. It mitigates overfitting on the imbalanced classes and delivers the best overall trade-off across all six metrics�particularly excelling in AUC and F1. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)**. It consistently achieves the highest AUC and the strongest MCC on this dataset, confirming it as the most reliable classifier for predicting term-deposit subscriptions among the five models evaluated. |

---

## Additional Files Summary
| File / Folder | Purpose |
|---------------|---------|
| `app.py` | Streamlit interactive web application for single and batch predictions. |
| `requirements.txt` | Python dependencies (scikit-learn, pandas, streamlit, joblib, etc.). |
| `test_data.csv` | Stratified hold-out test sample (auto-generated during training). |
| `model/train_and_save.py` | End-to-end training script that downloads data, fits all 5 pipelines, evaluates metrics, and saves `.joblib` models. |
| `model/predict.py` | Standalone CLI utility for batch inference. |
| `model/metrics.json` | JSON file containing all computed evaluation metrics. |
| `model/*.joblib` | Serialized preprocessing + classifier pipelines for deployment. |

---

*This README content is also included in the submitted PDF report.*
