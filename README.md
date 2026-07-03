# Bank Marketing — Term Deposit Subscription Prediction

End-to-end binary classification project predicting whether a bank client will subscribe to a term deposit, based on the [UCI Bank Marketing dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing) (~41k records of a Portuguese bank's telemarketing campaigns).

The focus of this project is **methodological rigor**: a fully leakage-free pipeline, honest evaluation against a baseline, and business-driven decision threshold optimization — not just chasing a single metric.

## Key Results

| Model | Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|---|
| Dummy Classifier (baseline) | 0.888 | 0.000 | 0.000 | 0.000 | 0.500 |
| Logistic Regression | 0.822 | 0.343 | 0.635 | 0.445 | 0.792 |
| Random Forest | 0.893 | 0.543 | 0.282 | 0.371 | 0.773 |
| Gradient Boosting | 0.845 | 0.386 | 0.636 | 0.480 | 0.804 |

*5-fold cross-validation on the training set (positive class: "yes")*

**Final model — tuned Gradient Boosting with optimized decision threshold (0.64), evaluated on the held-out test set:**

| Accuracy | Precision | Recall | F1 | AUC |
|---|---|---|---|---|
| **0.870** | **0.441** | **0.540** | **0.486** | **0.791** |

Threshold optimization alone improved test F1 from 0.456 → 0.486, trading a moderate drop in recall for a substantial precision gain (0.359 → 0.441) — fewer wasted calls to uninterested clients.

## Why This Project Is Non-Trivial

- **Severe class imbalance (~11% positive class).** Accuracy is misleading here — the dummy baseline reaches 88.8% accuracy with zero business value (F1 = 0). All modeling decisions were driven by F1/AUC, not accuracy.
- **Deliberate removal of the duration feature. Call duration is only known after the call ends, using it would be target leakage. It is excluded here to reflect a realistic pre-call prediction scenario.
- **Zero data leakage by design:**
  - All preprocessing (imputation, scaling, outlier capping, encoding) lives *inside* a scikit-learn `Pipeline` and is fitted only on training folds during cross-validation.
  - The decision threshold was selected on **out-of-fold predictions from the training set** — never on the test set. The test set was touched exactly once, at the very end.
  - Test F1 (0.486) landing close to the threshold-selection CV F1 (0.510) confirms the model generalizes well and is not overfitted.

## Workflow

### 1. Exploratory Data Analysis
- Uncovered hidden missing values — the dataset encodes them as the string `"unknown"` (up to 20.9% in the `default` column), invisible to `df.info()`.
- Correlation analysis revealed strong multicollinearity among macroeconomic indicators (`emp.var.rate` ↔ `euribor3m`: r = 0.97) → redundant features dropped to stabilize the linear model.
- Distribution analysis (histograms, boxplots) + outlier quantification using both **Z-score** (~7.5% of rows) and **IQR** (~20.5%) methods → decision to **cap outliers instead of removing** them, avoiding massive information loss.

### 2. Feature Engineering
- `pdays` (999 = never contacted) transformed into a binary `contacted_before` flag — the raw encoding made the variable numerically meaningless.
- Dropped `duration` (target leakage) and two collinear macroeconomic features.

### 3. Preprocessing Pipeline (`ColumnTransformer`)
- **Numerical branch:** custom `OutlierCapper` transformer (own implementation of `BaseEstimator` + `TransformerMixin`, IQR-based clipping) → median `SimpleImputer` → `StandardScaler`
- **Categorical branch:** most-frequent `SimpleImputer` → `OneHotEncoder(handle_unknown='ignore')`

### 4. Modeling & Evaluation
- Three algorithms representing distinct learning paradigms: **Logistic Regression** (interpretable linear baseline), **Random Forest** (bagging), **HistGradientBoostingClassifier** (boosting) — all with `class_weight='balanced'` to address imbalance.
- Compared against a **DummyClassifier** baseline to prove the models learn real patterns.
- Evaluated with 5-fold `cross_val_predict`: accuracy, precision, recall, F1, confusion matrices, ROC curves + AUC.
- Interesting finding: Random Forest had the *highest accuracy* but the *worst recall* (28%) — a textbook case of accuracy being deceptive under imbalance. Boosting handled the minority class markedly better.

### 5. Hyperparameter Tuning
- `GridSearchCV` (5-fold, custom F1 scorer via `make_scorer`) over `learning_rate`, `max_depth`, `max_iter` for the winning Gradient Boosting model.
- Best configuration: `learning_rate=0.1, max_depth=5, max_iter=100`.

### 6. Decision Threshold Optimization
- The default 0.5 threshold is arbitrary and ignores the business context: a wasted call (FP) costs less than a missed subscriber (FN), but call-center capacity is finite.
- Swept thresholds 0.10–0.90 on out-of-fold training predictions; optimal F1 at **0.64**.
- Final recommendation explicitly framed as a **business trade-off**: lower thresholds if coverage of interested clients is the priority, 0.64 if reducing unnecessary contacts matters more.

## Tech Stack

`Python` · `pandas` · `NumPy` · `scikit-learn` (Pipeline, ColumnTransformer, custom transformers, GridSearchCV, cross_val_predict) · `SciPy` · `Matplotlib` · Jupyter Notebook

## Repository Structure

```
├── Bank_Marketing.ipynb   # Full analysis: EDA → pipeline → modeling → tuning → threshold optimization
└── README.md
```

## How to Run

1. Download `bank-additional-full.csv` from the [UCI repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) and place it next to the notebook.
2. Install dependencies:
   ```bash
   pip install pandas numpy scikit-learn scipy matplotlib jupyter
   ```
3. Run the notebook top to bottom:
   ```bash
   jupyter notebook Bank_Marketing.ipynb
   ```

## Possible Extensions

- Cost-sensitive threshold selection using actual call cost vs. expected deposit value
- Probability calibration (`CalibratedClassifierCV`) before thresholding
- SHAP-based feature importance for model interpretability
- Comparison with XGBoost/LightGBM and resampling strategies (SMOTE)
