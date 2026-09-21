# 🛸 Commercial Drone Corridors - Flight Risk Classification

A machine learning project that analyzes commercial drone flight data across designated air corridors and builds a **Decision Tree Classifier** to predict the **risk level** (`Low`, `Medium`, `High`, `Critical`) of a given flight based on operational, environmental, and compliance features.

---

## 📌 Project Overview

Commercial drone delivery and logistics operators (e.g. Amazon Prime Air, Zipline, Wing, Matternet) fly thousands of missions through shared air corridors every year. Each flight carries a different combination of weather exposure, payload, airspace class, and operator behavior — all of which influence the operational risk of that flight.

This project:

1. Performs exploratory data analysis (EDA) on **12,000 simulated commercial drone flights**.
2. Visualizes univariate, bivariate, and multivariate relationships between flight conditions and risk.
3. Encodes and scales the data for machine learning.
4. Trains a **Decision Tree Classifier** to predict `risk_level`.
5. Tunes the model using **GridSearchCV** and **RandomizedSearchCV**.
6. Evaluates performance using accuracy, precision, recall, F1-score, cross-validation, residual analysis, and a confusion matrix.

---

## 🗂️ Repository Structure

```
commercial-drone-corridors-risk-classifier/
│
├── Commercial_Drone_Corridors_DT_Classifier.ipynb   # Main analysis & modeling notebook
├── commercial_drone_corridors.csv                   # Dataset (12,000 flight records)
├── README.md                                        # Project documentation (this file)
└── requirements.txt                                 # Python dependencies
```

---

## 📊 Dataset

**File:** `commercial_drone_corridors.csv`
**Rows:** 12,000 flights **Columns:** 23
**Date range:** Jan 2021 – Dec 2024

| Column | Description |
|---|---|
| `flight_id` | Unique identifier for each flight |
| `timestamp` | Date and time of the flight |
| `corridor_id` | ID of the designated air corridor (50 unique corridors) |
| `drone_type` | Multirotor, Fixed-Wing, Hybrid VTOL, Single-Rotor |
| `operator_name` | Operating company (Amazon Prime Air, Zipline, Wing, Matternet, etc. — 10 operators) |
| `airspace_class` | FAA airspace classification (B, C, D, E, G) |
| `cargo_type` | Retail Package, Mail, Medical Supplies, Emergency Response, etc. |
| `season` | Winter, Spring, Summer, Autumn |
| `weather_condition` | Clear, Overcast, Light Rain, Fog, Strong Wind, Partly Cloudy |
| `flight_status` | Completed, Aborted, Delayed, Diverted |
| `altitude_m` | Cruising altitude in meters |
| `ground_speed_kmh` | Ground speed in km/h |
| `flight_distance_km` | Total distance flown |
| `payload_kg` | Payload weight carried |
| `battery_start_pct` / `battery_end_pct` | Battery level at takeoff / landing |
| `wind_speed_kmh` | Wind speed during flight |
| `temperature_c` | Ambient temperature |
| `visibility_km` | Visibility range |
| `corridor_latitude` / `corridor_longitude` | Geographic coordinates of the corridor |
| `compliant_flight` | 1 = compliant with regulations, 0 = non-compliant |
| `risk_level` | **Target variable** — Low, Medium, High, Critical |

No missing values or duplicate rows were found in the dataset.

---

## ⚙️ Tech Stack

- **Language:** Python 3
- **Data handling:** `pandas`, `numpy`
- **Visualization:** `matplotlib`, `seaborn`
- **Machine Learning:** `scikit-learn`
  - `DecisionTreeClassifier`
  - `train_test_split`, `StandardScaler`, `LabelEncoder`
  - `GridSearchCV`, `RandomizedSearchCV`
  - `cross_val_score`
  - `classification_report`, `confusion_matrix`, `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`
- **Environment:** Jupyter Notebook

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/commercial-drone-corridors-risk-classifier.git
cd commercial-drone-corridors-risk-classifier
```

### 2. Create a virtual environment (optional but recommended)
```bash
python -m venv venv
source venv/bin/activate      # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Launch the notebook
```bash
jupyter notebook Commercial_Drone_Corridors_DT_Classifier.ipynb
```

Make sure `commercial_drone_corridors.csv` is in the same directory as the notebook before running the cells.

---

## 🧠 Methodology

1. **Data Loading & Cleaning** — Load the CSV, inspect shape/structure (`.info()`, `.describe()`), check for nulls and duplicates.
2. **Exploratory Data Analysis (18+ charts)**
   - Univariate categorical plots (risk level, flight status, drone type, airspace class, cargo type)
   - Univariate numerical plots (altitude, speed, distance, wind, temperature)
   - Bivariate/multivariate plots (battery usage, payload vs. speed, distance vs. status, risk vs. weather)
   - Correlation heatmap and compliance-vs-risk crosstab
   - Geographic scatter plot of corridors
3. **Preprocessing**
   - Label-encode all categorical columns
   - Separate features (`X`) and target (`y = risk_level`)
   - Standardize features with `StandardScaler`
   - 80/20 train-test split
4. **Modeling** — Train a baseline `DecisionTreeClassifier`.
5. **Evaluation** — Accuracy, classification report, 5-fold cross-validation.
6. **Hyperparameter Tuning** — `GridSearchCV` and `RandomizedSearchCV` over `criterion`, `max_depth`, `min_samples_split`, `min_samples_leaf`.
7. **Diagnostics** — Residual analysis on predicted probabilities and a confusion matrix for the final model.

---

## 📈 Results

| Metric | Value |
|---|---|
| Baseline Decision Tree Accuracy | ~26.7% |
| Mean 5-Fold CV Accuracy | ~25.6% (± 0.95%) |
| Best GridSearchCV Params | `criterion='entropy', max_depth=5, min_samples_leaf=2, min_samples_split=2` |
| Best RandomizedSearchCV Params | `criterion='entropy', max_depth=5, min_samples_leaf=1, min_samples_split=5` |

> **Note:** With four balanced-ish risk classes, random guessing would score ~25%. The current model performs only marginally above chance, suggesting that `risk_level` in this dataset is largely independent of the available features (consistent with a synthetically generated target). See **Future Improvements** below.

---

## 🔧 Future Improvements

- Add the missing `from sklearn.model_selection import GridSearchCV, RandomizedSearchCV` import at the top of the tuning section.
- Try ensemble methods (`RandomForestClassifier`, `XGBoost`, `GradientBoostingClassifier`) which typically outperform a single decision tree.
- Engineer new features (e.g., battery drop = `battery_start_pct - battery_end_pct`, time-of-day buckets from `timestamp`).
- Address class imbalance if present using `class_weight='balanced'` or SMOTE.
- Drop or hash high-cardinality identifier columns (`flight_id`, `timestamp`, `corridor_id`) which add noise rather than signal.
- Use `roc_auc_score` (already imported) with a one-vs-rest scheme for a fuller multiclass evaluation.

---
