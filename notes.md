# 📓 Detailed Notes — Breast Cancer ML Project

## 🏢 VAUTECH IT SOLUTIONS — TASK 3

| | |
|---|---|
| **Intern** | Kavya Jain |
| **Intern ID** | VT26ML005 |
| **Domain** | Machine Learning |
| **Company** | VAUTECH IT SOLUTIONS |
| **Mentor** | Vishal Rajbhar |

---

## 📌 Project Overview

This project performs complete **Data Preprocessing and Classification Model Building** on the UCI Breast Cancer Dataset. The goal is to predict whether a breast cancer patient will have a recurrence event or not, based on 9 medical features.

---

## 📂 Dataset Information

| Detail | Info |
|---|---|
| Source | UCI Machine Learning Repository |
| URL | https://archive.ics.uci.edu/dataset/14/breast+cancer |
| Instances | 286 (272 after cleaning) |
| Features | 9 + 1 target column |
| Task Type | Binary Classification |
| Target | `no-recurrence-events` / `recurrence-events` |
| Missing Values | Yes — denoted by `"?"` |

### Column Description

| Column | Type | Description | Missing? |
|---|---|---|---|
| Class | Binary (Target) | no-recurrence-events / recurrence-events | No |
| age | Categorical | Age group of patient (e.g. 30-39, 40-49) | No |
| menopause | Categorical | Menopause status (lt40, ge40, premeno) | No |
| tumor-size | Categorical | Size of tumor in mm ranges | No |
| inv-nodes | Categorical | Number of involved lymph nodes | No |
| node-caps | Binary | Whether cancer has spread to node capsule | **Yes (8)** |
| deg-malig | Integer | Degree of malignancy (1, 2, or 3) | No |
| breast | Binary | Which breast (left / right) | No |
| breast-quad | Categorical | Quadrant of the breast affected | **Yes (1)** |
| irradiat | Binary | Whether patient received radiation (yes / no) | No |

---

## 🔬 Step-by-Step Explanation

---

### ✅ Step 1 — Import Libraries

```python
import pandas as pd
import matplotlib.pyplot as plt
```

**What & Why:**
- `pandas` — the primary library for loading, manipulating, and analyzing tabular data (DataFrames).
- `matplotlib.pyplot` — used for creating visualizations like charts and plots.
- Other libraries like `numpy`, `seaborn`, and `sklearn` are imported later as needed. This is a clean practice — import only what you need, when you need it.

---

### ✅ Step 2 — Load the Dataset

```python
df = pd.read_csv('breast-cancer.data', header=None)
```

**What & Why:**
- `pd.read_csv()` reads the `.data` file (which is a comma-separated file) into a pandas DataFrame.
- `header=None` tells pandas that the file has **no header row** — the first row is actual data, not column names.
- Without `header=None`, pandas would treat the first patient's data as column names — which is wrong!

---

### ✅ Step 3 — Preview the Data

```python
df.head()
```

**What & Why:**
- `df.head()` displays the **first 5 rows** of the DataFrame.
- It's a quick sanity check to confirm the data loaded correctly.
- You can pass a number like `df.head(10)` to see the first 10 rows.
- At this stage, columns show as `0, 1, 2...` because no column names were assigned yet.

---

### ✅ Step 4 — Assign Column Names

```python
df.columns = ('Class', 'age', 'menopause', 'tumor-size', 'inv-nodes', 
              'node-caps', 'deg-malig', 'breast', 'breast-quad', 'irradiat')
```

**What & Why:**
- Column names were found in the `breast-cancer.names` documentation file provided with the dataset.
- Assigning proper names makes the data readable and easier to work with.
- Without proper names, we would have to reference columns as `df[0]`, `df[1]` etc. — which is confusing.

---

### ✅ Step 5 — Check Dataset Shape

```python
df.shape
# Output: (286, 10)
```

**What & Why:**
- `df.shape` returns a tuple `(rows, columns)`.
- Output `(286, 10)` means **286 patients** and **10 columns** (9 features + 1 target).
- This matches the UCI documentation — confirming the dataset loaded correctly.

---

### ✅ Step 6 — Dataset Info

```python
df.info()
```

**What & Why:**
- `df.info()` gives a summary of the DataFrame including:
  - Number of rows and columns
  - Column names
  - Non-null count (how many values are not missing)
  - Data type of each column (`object` = text/string, `int64` = integer)
- **Important observation:** All columns show `286 non-null` — but this is misleading! Missing values in this dataset are stored as `"?"` strings, not as actual `NaN`. So pandas doesn't detect them yet.
- `object` dtype means the column contains text — ML models cannot work with text directly. Everything must be converted to numbers later.

---

### ✅ Step 7 — Check for Missing Values (First Check)

```python
df.isnull().sum()
```

**What & Why:**
- `df.isnull()` creates a boolean DataFrame — `True` where values are missing, `False` otherwise.
- `.sum()` counts the `True` values per column.
- **Result shows all zeros** — because missing values are stored as `"?"` strings, not as `NaN`.
- This is a classic real-world data problem — missing values disguised as strings!

---

### ✅ Step 8 — Replace "?" with NaN

```python
import numpy as np
df.replace('?', np.nan, inplace=True)
```

**What & Why:**
- `np.nan` is the standard Python representation of a missing/null value.
- `df.replace('?', np.nan)` finds all `"?"` strings and replaces them with actual `NaN`.
- `inplace=True` modifies the original DataFrame directly instead of creating a new one.
- After this step, `df.isnull().sum()` will correctly detect the missing values.

---

### ✅ Step 9 — Check Missing Values (Second Check)

```python
df.isnull().sum()
```

**Output:**
```
node-caps      8
breast-quad    1
(all others)   0
```

**What & Why:**
- Now we can see the **real** missing values:
  - `node-caps` — 8 missing values
  - `breast-quad` — 1 missing value
- This matches exactly what the `breast-cancer.names` documentation stated.
- These need to be filled before we can train a model.

---

### ✅ Step 10 — Fill Missing Values with Mode

```python
df['node-caps'] = df['node-caps'].fillna(df['node-caps'].mode()[0])
df['breast-quad'] = df['breast-quad'].fillna(df['breast-quad'].mode()[0])
```

**What & Why:**
- Both `node-caps` and `breast-quad` are **categorical columns** (text values).
- For categorical columns, we fill missing values with the **mode** — the most frequently occurring value.
- We cannot use mean or median for text columns — those are only for numerical data.
- `df['column'].mode()[0]` gets the most frequent value. `[0]` is used because mode() returns a Series — we take the first (most frequent) value.
- `fillna()` fills all `NaN` values with the given value.
- We assign back with `df['column'] = ...` to ensure the change is saved.

---

### ✅ Step 11 — Verify No Missing Values Remain

```python
df.isnull().sum()
# All columns now show 0
```

**What & Why:**
- Run `isnull().sum()` again to confirm all missing values have been filled.
- All columns should now show `0` — confirming the data is clean.

---

### ✅ Step 12 — Remove Duplicate Rows

```python
df = df.drop_duplicates()
df.shape
# Output: (272, 10)
```

**What & Why:**
- Duplicate rows are identical entries that appear more than once in the dataset.
- They can bias the model — if a pattern appears twice, the model may learn it as more important than it actually is.
- `drop_duplicates()` removes all rows that are exact copies of another row.
- We use `df = df.drop_duplicates()` to save the result back to `df`. Without this assignment, the original `df` would remain unchanged.
- **Result:** 286 - 14 duplicates = **272 clean rows** remaining.

---

### ✅ Step 13 — Label Encoding (Encode Categorical Features)

```python
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in df.columns:
    df[col] = le.fit_transform(df[col])

df.head()
```

**What & Why:**
- ML algorithms only understand **numbers** — not text like "yes", "no", "left", "right".
- `LabelEncoder` converts each unique text value in a column to a unique integer.
  - Example: `no → 0`, `yes → 1`
  - Example: `ge40 → 0`, `lt40 → 1`, `premeno → 2`
- We loop through **all columns** including the target (`Class`) because it also contains text (`no-recurrence-events`, `recurrence-events`).
- `fit_transform()` first learns the mapping then applies it in one step.
- After encoding: `no-recurrence-events → 0`, `recurrence-events → 1`

---

### ✅ Step 14 — Feature Scaling (StandardScaler)

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = df.drop('Class', axis=1)
y = df['Class']

X_scaled = scaler.fit_transform(X)
print(X_scaled[:5])
```

**What & Why:**
- **Scaling** brings all feature values to the same range so no single feature dominates the model due to its larger numerical range.
- `StandardScaler` transforms each feature so that it has **mean = 0** and **standard deviation = 1**.
- We separate features (`X`) and target (`y`) before scaling.
  - `X = df.drop('Class', axis=1)` — all columns except the target.
  - `y = df['Class']` — only the target column.
- **Why not scale the target `y`?** Because `y` contains class labels (0 and 1) that the model needs to predict. Scaling them would turn `0` and `1` into decimals which breaks classification.
- `fit_transform()` calculates the mean and std of each feature, then scales it — all in one step.

---

### ✅ Step 15 — Train-Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

print("Training size:", X_train.shape)  # (217, 9)
print("Testing size:", X_test.shape)    # (55, 9)
```

**What & Why:**
- We split the data into two parts:
  - **Training set (80%)** — used to train/teach the model.
  - **Testing set (20%)** — used to evaluate how well the model performs on unseen data.
- `test_size=0.2` means 20% of data goes to testing.
- `random_state=42` ensures the split is the same every time you run the code (reproducibility). Without it, you'd get a different split each time.
- **Result:** 217 rows for training, 55 rows for testing.

---

### ✅ Step 16 — Build the Model (Logistic Regression)

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)

print("Model Training completed")
```

**What & Why:**
- **Logistic Regression** is a classification algorithm used to predict binary outcomes (0 or 1).
- Despite its name, it's a **classification** algorithm, not regression.
- It works by calculating the probability that a sample belongs to a class (e.g., 70% chance of recurrence).
- `model.fit(X_train, y_train)` trains the model — it learns the patterns from the training data.
- It's a great starting point for classification problems — simple, fast, and interpretable.

---

### ✅ Step 17 — Make Predictions

```python
y_pred = model.predict(X_test)
print(y_pred)
```

**What & Why:**
- `model.predict(X_test)` uses the trained model to predict the class labels for the test data.
- It returns an array of `0`s and `1`s — where `0 = no-recurrence` and `1 = recurrence`.
- These predictions are then compared against the actual labels (`y_test`) to evaluate model performance.

---

### ✅ Step 18 — Evaluate with Accuracy Score

```python
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)
print("Model Accuracy", accuracy * 100, "%")
# Output: 72.72%
```

**What & Why:**
- `accuracy_score` compares predicted labels (`y_pred`) vs actual labels (`y_test`).
- It calculates the percentage of correct predictions.
- **72.72% accuracy** means the model correctly predicted 72.72% of the test patients.
- This is a good result — published research papers on this same dataset report accuracy in the range of **66% - 78%**.

---

### ✅ Step 19 — Confusion Matrix

```python
import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()
```

**What & Why:**
- A **Confusion Matrix** shows a detailed breakdown of correct and incorrect predictions.
- It has 4 values:
  - **True Positive (TP)** — Model predicted recurrence, actually recurrence ✅
  - **True Negative (TN)** — Model predicted no-recurrence, actually no-recurrence ✅
  - **False Positive (FP)** — Model predicted recurrence, but actually no-recurrence ❌
  - **False Negative (FN)** — Model predicted no-recurrence, but actually recurrence ❌
- `sns.heatmap()` visualizes the matrix as a colored grid — easier to read.
- `annot=True` shows the actual numbers inside each cell.
- `fmt='d'` formats numbers as integers (not decimals).

---

## 📊 Final Results

| Metric | Value |
|---|---|
| Training Size | 217 rows |
| Testing Size | 55 rows |
| Model Used | Logistic Regression |
| Accuracy | **72.72%** |

> Published research on this dataset reports 66% - 78% accuracy. Our result is well within that range! ✅

---

## 🛠️ Libraries Used

| Library | Purpose |
|---|---|
| `pandas` | Data loading, cleaning, manipulation |
| `numpy` | Replacing `"?"` with `NaN` |
| `matplotlib` | Visualization (Confusion Matrix plot) |
| `seaborn` | Enhanced visualization (Heatmap) |
| `sklearn.preprocessing.LabelEncoder` | Encoding categorical columns |
| `sklearn.preprocessing.StandardScaler` | Feature scaling |
| `sklearn.model_selection.train_test_split` | Splitting data |
| `sklearn.linear_model.LogisticRegression` | Classification model |
| `sklearn.metrics.accuracy_score` | Evaluating accuracy |
| `sklearn.metrics.confusion_matrix` | Detailed prediction breakdown |

---

## 📜 Citation

Zwitter, M. & Soklic, M. (1988). Breast Cancer [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51P4M

---

*Notes prepared as part of ML Internship Task 3 — VAUTECH IT SOLUTIONS*
