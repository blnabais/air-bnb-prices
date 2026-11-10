# Airbnb price prediction in Hong Kong

**Linear and logistic regression implemented from scratch, with gradient descent and no machine learning libraries.**

![Python](https://img.shields.io/badge/Python-3.11+-blue)
![NumPy](https://img.shields.io/badge/NumPy-only-013243)
![No ML libraries](https://img.shields.io/badge/scikit--learn-not%20used-critical)

Two models over 6,046 Airbnb listings: one predicts the nightly price, the other
flags the listings in the top 20% of price within comparable groups. All of the
maths — gradient descent, L2 regularisation, sigmoid, k-fold — is written by hand
in `linear_regression.py` and `logistic_regression.py`.

| | Metric | Test |
|---|---|---|
| **Regression** (log-price) | R² | **0.6746** |
| | RMSE | 0.4989 |
| **Classification** (top 20%) | ROC-AUC | **0.8043** |
| | F1 (threshold 0.273) | 0.4891 |

<p align="center">
  <img src="figures/preco_vs_log_preco.png" width="48%" alt="Price and log-price distributions">
  <img src="figures/curva_roc.png" width="48%" alt="ROC curve of the classifier">
</p>

### What is worth looking at here

- **Nothing is a black box.** Both regressions are my own code: initialisation,
  gradient, weight updates, stopping criterion.
- **Every preprocessing decision is tested, not assumed.** The ablation table in
  `report/ablacoes.csv` compares imputation and outlier strategies side by side —
  and shows that two of them made no difference at all.
- **Taking the log of the price is the decision that carries the project:** R²
  goes from 0.188 to 0.660 just by changing the target.
- **The threshold was tuned, not left at 0.5:** recall goes from 0.165 to 0.574.

**Author:** Bernardo Nabais · **Course unit:** Machine Learning · **Degree:**
Artificial Intelligence and Data Science, University of Beira Interior ·
**Academic year:** 2026/2027

---

## 1. Goal

A complete supervised learning pipeline over Airbnb listings, with two
objectives:

- **Regression:** predict the nightly price from the listing's characteristics.
- **Classification:** identify listings in the top 20% of price within comparable groups.

Every model is implemented from scratch with gradient descent, without machine
learning libraries. The focus is on justifying the preprocessing decisions and on
interpreting the models.

---

## 2. Dataset

| | |
|---|---|
| **City** | Hong Kong |
| **Source** | [Inside Airbnb - Get the Data](https://insideairbnb.com/get-the-data/) |
| **Scraping dates** | 2026-06-28, 2026-06-29, 2026-07-03 |
| **Currency** | HKD (Hong Kong dollars) |
| **Original size** | 6,734 rows × 90 columns |
| **After cleaning** | 6,046 rows × 51 features |

The downloaded file (`listings.csv.gz`) was decompressed and placed in
`data/listings.csv`. The code reads that path directly; `pd.read_csv` accepts
either form, you only need to adjust the name.

---

## 3. Project layout

```text
projeto1_ml_a55903/
├── data/
│   ├── listings.csv                # Original dataset (decompressed)
│   └── processado/                 # Final matrices produced by the pipeline
├── figures/                       # Figures produced by the notebook
├── report/
│   ├── ablacoes.csv               # Comparison table of the strategies
│   ├── coeficientes_linear.csv
│   └── coeficientes_logistica.csv
├── exploracao.ipynb                # Exploratory analysis and experiments
├── data_preparation.py             # Preprocessing pipeline
├── linear_regression.py            # Linear regression (gradient descent + L2)
├── logistic_regression.py          # Logistic regression (gradient descent)
└── README.md
```

---

## 4. How to reproduce

**Requirements:** Python 3.11+, with `pandas`, `numpy` and `matplotlib`.
No machine learning library is used.

```bash
pip install pandas numpy matplotlib jupyter
```

**Steps:**

1. Download the Hong Kong `listings.csv.gz` from
   [insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data/),
   decompress it and place it in `data/listings.csv`.
2. Open `exploracao.ipynb` and run all the cells in order
   (*Restart & Run All*).
3. Figures are written to `figures/`, tables to `report/`, and the final
   matrices to `data/processado/`.

**Reproducibility:** every random split uses `seed=42`
(train/validation/test split and k-fold). The results are deterministic.

**Using the pipeline directly:**

```python
from data_preparation import pipeline_completo

X_treino, (X_val, X_teste), y_treino, (y_val, y_teste) = pipeline_completo(
    estrategia_missing="simples",      # or "agrupada"
    estrategia_outliers="nenhuma",     # or "capping" / "filtragem"
    polinomiais=True,
    interacoes=True,
    guardar=True,
)
```

---

## 5. Preprocessing decisions

### 5.1 Target

- The `price` column, converted from text (`"$1,181.02"`) to a number.
- Confirmed to be the nightly price (equal to `price_quote_price_per_night`).
- **688 rows with no price removed** (10.2%) — the target is never imputed.
- Model target: **log(price)**, because the distribution is heavily skewed
  (median 424 HKD, maximum 171,234 HKD).

### 5.2 Data leakage

Excluded from the features because they are derived from the target:
`price_quote_total_price`, `price_quote_price_per_night`,
`estimated_revenue_l365d`, `price_quote_raw`.

### 5.3 Simulated length of stay

- **Decision:** keep every length of stay and use the length as a feature.
- **Justification:** the median drops from 545 HKD (1 night) to 221 HKD (28-31
  nights). In Hong Kong, stays shorter than 28 nights require a licence, which
  explains why listings cluster in the 28-31 group (2,535 listings).
- **Features created:** `noites` (numeric) and `long_stay` (binary, ≥ 28).
- **Validating the decision:** residual bias is close to zero across every
  length-of-stay group (between −0.004 and +0.059 for 1 to 31 nights).

### 5.4 Missing values

**Missing with a meaning:**
- Review columns (~45%, correlation 1.00 with each other — they are missing on
  the same rows): created a `tem_reviews` feature; `review_scores_rating` imputed
  with the median of the listings that do have reviews.

**Unknown value — two policies compared:**
- `bedrooms` (44.8%), `bathrooms` (11%), `beds` (4.9%).
- **Simple** policy: global median of the training set.
- **Grouped** policy: median per `room_type × accommodates` group.
- `bathrooms_text` (0.3% missing) used to recover `bathrooms` values before any
  imputation.

**Removed:**
- 14 columns that are 100% empty, including `host_response_rate`,
  `host_response_time`, `host_acceptance_rate`, `neighbourhood`, `license` and
  `instant_bookable`. A limitation of this version of the file.
- `property_type` (55 categories), removed because of complete multicollinearity
  with `room_type` — see section 5.7.

### 5.5 Outliers

Two robust criteria applied to the log-price:
- **Tukey's IQR (1.5 × IQR):** 40 outliers (0.66%), bounds [29, 6,823] HKD
- **MAD (robust z > 3):** 28 outliers (0.46%), bounds [21, 8,485] HKD

Three strategies compared: no treatment, capping (winsorising at the IQR bounds)
and robust filtering (removing the union IQR ∪ MAD).

**`minimum_nights` and `accommodates`** were analysed with the same criteria. For
`minimum_nights`, MAD flags 43% of the observations because the distribution is
bimodal (the unimodality assumption does not hold); both variables were kept
untreated.

**Methodological note:** outlier treatment is applied **only to the training
set**. Winsorising the target on validation and test would produce optimistic
metrics that are not comparable across strategies.

### 5.6 Amenities

871 distinct values, turned into:
- `n_amenities`: total count per listing.
- One **binary feature for each of the 15 most frequent amenities**.

The top 15 are selected over the full dataset, before the split. Since it does
not use the price, the leakage risk is negligible.

### 5.7 Multicollinearity

An early version included `room_type` and `property_type` at the same time. A
contingency table showed complete redundancy (each `property_type` category maps
exclusively to one `room_type`), and the coefficients had signs that contradicted
the exploratory analysis. `property_type` was removed, reducing the matrix from
109 to 51 features.

### 5.8 Split, validation and leakage prevention

- **70% train / 15% validation / 15% test** split, with `seed=42`.
- **5-fold cross-validation** to compare models and select hyperparameters.
- Every preprocessing step that learns parameters from the data (imputation
  medians, outlier bounds, mean and standard deviation for standardisation,
  positive-class thresholds) is computed **on the training set only** and applied
  to the other sets. Inside the k-folds, that learning happens within each fold.
- The test set is used **only for the final report**, never to make decisions.

### 5.9 Final features (51)

- **Numeric:** `accommodates`, `bedrooms`, `bathrooms`, `beds`,
  `minimum_nights`, `noites`, `n_amenities`, `long_stay`,
  `review_scores_rating`, `number_of_reviews`
- **Polynomial:** `accommodates²`, `n_amenities²` (built **before**
  standardisation)
- **Binary:** `tem_reviews` + 15 amenities
- **Categorical (one-hot):** `room_type`, `neighbourhood_cleansed`
- **Interactions:** `long_stay × room_type`

---

## 6. Results

### Regression

Final configuration: log-price, polynomial features and interactions, simple
imputation, no outlier treatment, L2 = 0.01.

| Set | MAE | RMSE | R² |
|---|---|---|---|
| Train | 0.3441 | 0.5014 | 0.6808 |
| Validation | 0.3560 | 0.5107 | 0.6908 |
| **Test** | **0.3609** | **0.4989** | **0.6746** |

Comparison of targets and features (mean of 5 folds):

| Model | R² |
|---|---|
| Raw price (HKD) | 0.188 |
| Log-price | 0.660 |
| Log + polynomial | 0.667 |
| Log + polynomial + interactions | 0.672 |

L2 regularisation (U-shaped curve, optimum at 0.01):

| λ | 0.000 | 0.001 | **0.010** | 0.100 | 1.000 |
|---|---|---|---|---|---|
| R² | 0.672 | 0.673 | **0.677** | 0.671 | 0.585 |

### Classification

Positive class: top 20% of price within `room_type × neighbourhood`, with
thresholds learned on the training set.

| Metric | Threshold 0.5 | Threshold 0.273 |
|---|---|---|
| Accuracy | 0.8282 | 0.7676 |
| Precision | 0.7632 | 0.4262 |
| Recall | 0.1648 | **0.5739** |
| F1 | 0.2710 | **0.4891** |
| ROC-AUC | 0.8043 | 0.8043 |

The 0.273 threshold was selected by maximising F1 on the validation set and
applied to the test set with no further tuning.

<p align="center">
  <img src="figures/residuos.png" width="70%" alt="Residual analysis of the regression model">
</p>

### Main conclusions

**Decisions with the most impact:**
1. Log-price (R² from 0.188 to 0.660, and much more stable training).
2. Removing `property_type` (multicollinearity).
3. Using length of stay as a feature instead of filtering the dataset.
4. Tuning the classification threshold (recall from 0.165 to 0.574).

**Decisions with little impact:**
1. Outlier treatment — none of the three strategies made a relevant difference,
   probably because taking the log already compresses the extremes.
2. Imputation policy — simple and grouped are equivalent, because the imputed
   variables are redundant given `accommodates`.

---

## 7. Limitations

- Several columns required by the assignment are 100% empty in this version of
  the file (`host_response_rate`, `host_response_time`, `host_acceptance_rate`).
- Categories with few observations (Hotel room: 15, Shared room: 116) show high
  error and systematic bias; the model is calibrated for the dominant categories.
- The geographic coordinates are not used directly — the spatial effect is
  captured only at neighbourhood level.
- The coefficients describe associations, not causal relationships. The case of
  `number_of_reviews` (negative coefficient in both models) illustrates this: the
  plausible reading is that cheaper listings have higher occupancy and accumulate
  more reviews.
