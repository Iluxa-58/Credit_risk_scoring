# Credit Risk Scoring

Machine learning project for predicting the probability of loan default using anonymized customer application data from the Home Credit Default Risk dataset.

The project covers the complete modeling workflow: exploratory data analysis, leakage-safe preprocessing, baseline modeling, gradient boosting, probability evaluation, threshold optimization, and final testing on previously unseen data.

## Project Overview

Financial institutions need to estimate the probability that a borrower will fail to repay a loan. This is an imbalanced binary classification problem in which different prediction errors have different business consequences:

- a false negative may result in issuing a loan to a high-risk borrower;
- a false positive may result in rejecting a reliable customer.

The objective of this project is not only to maximize predictive performance but also to select a decision threshold that reflects the asymmetric cost of these errors.

## Key Results

CatBoost achieved the strongest performance on the validation set:

| Model | ROC-AUC | PR-AUC | Brier score |
|---|---:|---:|---:|
| Dummy baseline | 0.5000 | 0.0801 | 0.0742 |
| Logistic Regression | 0.7130 | 0.2139 | 0.0696 |
| CatBoost | **0.7603** | **0.2484** | **0.0676** |

Compared with Logistic Regression, CatBoost:

- increased ROC-AUC by **0.0473**;
- increased PR-AUC by **0.0345**;
- reduced Brier score by **0.0020**.

A lower Brier score indicates more accurate probability estimates. Final model selection and threshold optimization were performed using the validation set, while the test set remained untouched until the final evaluation.

## Final Test Results

| Metric | Result |
|---|---:|
| ROC-AUC | **[add after Stage 4]** |
| PR-AUC | **[add after Stage 4]** |
| Brier score | **[add after Stage 4]** |
| Precision | **[add after Stage 4]** |
| Recall | **[add after Stage 4]** |
| F1-score | **[add after Stage 4]** |
| Selected threshold | **[add after Stage 4]** |

## Dataset

The project uses `application_train.csv` from the [Home Credit Default Risk](https://www.kaggle.com/competitions/home-credit-default-risk) Kaggle competition.

Dataset characteristics:

- 307,511 loan applications;
- 122 columns;
- 120 potential model features after excluding the target and customer identifier;
- binary target: `TARGET = 1` represents a client with payment difficulties;
- positive-class rate of approximately 8.07%;
- 67 features containing missing values;
- 41 features with more than 50% missing values.

The dataset is not included in this repository. Download `application_train.csv` from Kaggle and place it in:

```text
data/application_train.csv
```

## Project Workflow

### 1. Exploratory Data Analysis

The first stage covers:

- dataset structure and data types;
- class imbalance;
- missing-value analysis;
- duplicate detection;
- target-rate comparison across customer groups;
- detection of the special `DAYS_EMPLOYED = 365243` value;
- identification of possible data leakage risks.

Key findings:

- the target class is strongly imbalanced;
- no complete duplicate rows were found;
- `SK_ID_CURR` is a unique identifier and was excluded from modeling;
- the special value in `DAYS_EMPLOYED` was treated as missing;
- observed relationships were interpreted as associations rather than causal effects.

Notebook: [`notebooks/01_eda.ipynb`](notebooks/01_eda.ipynb)

### 2. Baseline Modeling

The data was split into train, validation, and test sets using stratification.

All preprocessing operations were fitted exclusively on the training data to prevent leakage.

Two baseline models were evaluated:

- `DummyClassifier` using the class prior;
- Logistic Regression with separate preprocessing for numerical and categorical features.

Numerical preprocessing:

- median imputation;
- standardization.

Categorical preprocessing:

- most-frequent-value imputation;
- One-Hot Encoding with support for previously unseen categories.

Notebook: [`notebooks/02_baseline.ipynb`](notebooks/02_baseline.ipynb)

### 3. Gradient Boosting

CatBoost was selected because it can handle:

- nonlinear relationships;
- numerical missing values;
- categorical features without One-Hot Encoding;
- interactions between customer and loan characteristics.

Early stopping was used to reduce overfitting. Additional experiments with tree depths of 5 and 7 did not improve validation performance over the main model with depth 6.

Notebook: [`notebooks/03_boosting.ipynb`](notebooks/03_boosting.ipynb)

### 4. Business-Oriented Evaluation

The final stage includes:

- probability calibration analysis;
- threshold comparison;
- precision, recall, and F1-score evaluation;
- asymmetric error-cost analysis;
- confusion-matrix analysis;
- one-time evaluation on the untouched test set.

For demonstration purposes, a false negative is assigned a higher cost than a false positive. These costs are illustrative and do not represent the actual economics of a financial institution.

Notebook: [`notebooks/04_business_analysis.ipynb`](notebooks/04_business_analysis.ipynb)

## Feature Importance

The most important CatBoost features were:

1. `EXT_SOURCE_3`
2. `EXT_SOURCE_2`
3. `EXT_SOURCE_1`
4. `AMT_CREDIT`
5. `AMT_GOODS_PRICE`
6. `DAYS_BIRTH`
7. `DAYS_EMPLOYED`
8. `AMT_ANNUITY`

External credit scores contributed the most to the model’s predictions. Loan amount, goods price, customer age, employment history, and annuity amount were also important.

Feature importance describes how strongly the model uses a feature. It does not establish causality or indicate the direction of a feature’s effect.

![CatBoost feature importance](reports/figures/feature_importance.png)

## Project Structure

```text
credit-risk-scoring/
├── data/
│   └── application_train.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_baseline.ipynb
│   ├── 03_boosting.ipynb
│   └── 04_business_analysis.ipynb
├── reports/
│   └── figures/
│       ├── model_comparison.png
│       ├── feature_importance.png
│       ├── calibration_curve.png
│       └── threshold_cost.png
├── README.md
├── requirements.txt
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/USERNAME/credit-risk-scoring.git
cd credit-risk-scoring
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```bash
.venv\Scripts\activate
```

Install the dependencies:

```bash
pip install -r requirements.txt
```

Place `application_train.csv` in the `data/` directory and run the notebooks in order:

```text
01_eda.ipynb
02_baseline.ipynb
03_boosting.ipynb
04_business_analysis.ipynb
```

## Technologies

- Python
- pandas
- NumPy
- scikit-learn
- CatBoost
- Matplotlib
- Seaborn
- Jupyter Notebook

## Evaluation Metrics

Because only approximately 8% of applications belong to the positive class, accuracy is not used as the primary metric.

The project uses:

- **ROC-AUC** to measure ranking quality across classification thresholds;
- **PR-AUC** to evaluate positive-class detection under class imbalance;
- **Brier score** to assess probability accuracy;
- **precision and recall** to analyze threshold-dependent decisions;
- **business cost** to compare the consequences of false positives and false negatives.

## Limitations

- The project currently uses only `application_train.csv`.
- Business costs are illustrative and are not based on real financial data.
- Feature importance does not establish causal relationships.
- Historical patterns may change over time due to data drift.
- A production model would require fairness analysis, monitoring, explainability controls, and periodic retraining.
- The dataset is anonymized and does not represent a live lending decision system.

## Possible Improvements

Future work may include:

- integrating `bureau.csv` and other relational Home Credit tables;
- adding aggregated credit-history features;
- using SHAP for local and global model interpretation;
- probability calibration using an additional calibration split;
- monitoring data and prediction drift;
- implementing a reproducible training pipeline;
- exposing the model through a simple API.

## Author

**Ilya Klimashin**

HSE University student focused on Data Science, Machine Learning, and Data Analytics.

- GitHub: [Iluxa-58]
- LinkedIn: [add LinkedIn profile]