# 🎗️ Breast Cancer Prediction - ML Preprocessing Pipeline

##  VAUTECH IT SOLUTIONS - TASK 3

| | |
|---|---|
| **Intern** | Kavya Jain |
| **Intern ID** | VT26ML005 |
| **Domain** | Machine Learning |
| **Company** | VAUTECH IT SOLUTIONS |
| **Mentor** | Vishal Rajbhar |

---

## 📂 Dataset

- **Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/14/breast+cancer)
- **Instances:** 286 (272 after removing duplicates)
- **Features:** 9 + 1 target column
- **Task Type:** Binary Classification
- **Target:** `no-recurrence-events` / `recurrence-events`

---

## 🔁 Project Pipeline

### Step 1: Load Dataset
- Loaded dataset from UCI ML Repository
- Assigned correct column names from documentation

### Step 2: Explore Data
- Checked shape, dtypes and info
- Identified missing values hidden as `"?"`

### Step 3: Handle Missing Values
- Replaced `"?"` with `NaN`
- Found 8 missing values in `node-caps` and 1 in `breast-quad`
- Filled missing values with column mode

### Step 4: Remove Duplicates
- Found and removed 14 duplicate rows
- Dataset reduced from 286 → 272 rows

### Step 5: Encode Categorical Features
- Applied Label Encoding to all categorical columns
- Converted all text values to numerical integers

### Step 6: Feature Scaling
- Separated features (X) and target (y)
- Applied StandardScaler to all feature columns
- Target column left untouched

### Step 7: Train-Test Split
- Split data into 80% training and 20% testing
- Training size: 217 rows
- Testing size: 55 rows

### Step 8: Model Building
- Built Logistic Regression classification model
- Trained on X_train and y_train

### Step 9: Model Evaluation
- Predicted on X_test
- Evaluated using accuracy score, confusion matrix and classification report

---

## 📊 Results

| Metric | Score |
|--------|-------|
| Accuracy | 72.72% |

> Note: Research papers on this same dataset report accuracy in the range of 66% - 78% — our result is well aligned with published research!

---

## 🛠️ Tools & Libraries

| Tool | Purpose |
|------|---------|
| Python | Programming language |
| pandas | Data loading and manipulation |
| numpy | Numerical operations |
| scikit-learn | Encoding, scaling, splitting, model building |

---

## 📁 Files

| File | Description |
|------|-------------|
| `Breast_Cancer_ML_Internship.ipynb` | Main notebook with complete pipeline |
| `breast-cancer.data` | Raw dataset file |
| `breast-cancer.names` | Dataset documentation |

---

## 🚀 How to Run

1. Clone the repository
```bash
git clone https://github.com/yourusername/breast-cancer-ml-preprocessing.git
```

2. Install required libraries
```bash
pip install pandas numpy scikit-learn
```

3. Open the notebook
```bash
jupyter notebook Breast_Cancer_ML_Internship.ipynb
```

---

## 📜 Citation

Zwitter, M. & Soklic, M. (1988). Breast Cancer [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C51P4M

---

## 👨‍💻 Author

Kavya jain
