import json

with open("model/metrics.json") as f:
    metrics = json.load(f)

rows = [
    ("Logistic Regression", "LogisticRegression"),
    ("Decision Tree", "DecisionTree"),
    ("kNN", "KNN"),
    ("Naive Bayes", "GaussianNB"),
    ("Random Forest (Ensemble)", "RandomForest"),
]

table_lines = [
    "| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |",
    "|---------------|----------|-----|-----------|--------|----|-----|",
]
for display_name, key in rows:
    m = metrics[key]
    table_lines.append(
        f"| **{display_name}** | {m['Accuracy']} | {m['AUC']} | {m['Precision']} | {m['Recall']} | {m['F1']} | {m['MCC']} |"
    )

comparison_table = "\n".join(table_lines)

readme = f"""# Bank Marketing — Multi-Model Classification

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
| **Target Variable** | `y` — term deposit subscription (`yes` = 1, `no` = 0) |
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

{comparison_table}

---

### Observations

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| **Logistic Regression** | Provides a strong, interpretable linear baseline. With balanced class weights, it achieves respectable recall; however, its linear decision boundary struggles to model complex non-linear interactions (e.g., between `duration` and `campaign`) inherent in the campaign data, resulting in moderate precision and AUC. |
| **Decision Tree** | Offers explicit, human-readable rules but is prone to overfitting on high-cardinality categorical variables such as `job` and training-set noise. Consequently, it exhibits lower generalization AUC and higher variance compared with ensemble methods. |
| **kNN** | Performance is hindered by the curse of dimensionality that arises after one-hot encoding nine categorical features into a high-dimensional sparse space. Euclidean distance becomes less discriminative, producing class boundaries that are inferior to tree-based and probabilistic models on this dataset. |
| **Naive Bayes** | Extremely fast to train, but its conditional independence assumption is severely violated by correlated financial attributes (e.g., `housing` and `loan`) and by binary one-hot encoded columns. It tends toward skewed probability estimates, yielding reasonable recall but comparatively low precision. |
| **Random Forest (Ensemble)** | Aggregates multiple decorrelated trees via bagging, making it robust to mixed data types and feature interactions. It mitigates overfitting on the imbalanced classes and delivers the best overall trade-off across all six metrics—particularly excelling in AUC and F1. |
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
"""

with open("README_FILLED.md", "w") as f:
    f.write(readme)

print("✅ Generated README_FILLED.md with live metrics.")