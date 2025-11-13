# Regression Trees: NYC Taxi Tip Prediction

## Overview
This notebook demonstrates how to build and evaluate a **Decision Tree Regressor** model using real NYC taxi trip data. The goal is to predict the `tip_amount` based on other trip characteristics like fare, distance, and payment method.

---

## Workflow & Key Functions

### 1. **Import Libraries**
```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import normalize
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
```
- **pandas**: Data manipulation and analysis
- **numpy**: Numerical operations
- **sklearn**: Machine learning algorithms and utilities

---

### 2. **Dataset Analysis**

#### Load Data
```python
raw_data = pd.read_csv(url)
```
Reads the CSV file containing 13 taxi trip variables (fare_amount, tip_amount, distance, etc.).

#### Correlation Analysis
```python
correlation_values = raw_data.corr()['tip_amount'].drop('tip_amount')
correlation_values.plot(kind='barh', figsize=(10, 6))
```
- **Purpose**: Identify which features are most/least related to tip_amount
- **Output**: Horizontal bar chart showing correlation coefficients
- **Finding**: Features like `payment_type`, `VendorID`, and `improvement_surcharge` have minimal correlation and can be dropped

---

### 3. **Data Preprocessing**

#### Extract Features and Target
```python
y = raw_data[['tip_amount']].values.astype('float32')  # Target variable
X = raw_data.drop(['tip_amount'], axis=1).values       # Feature matrix
```

#### Normalize Features
```python
X = normalize(X, axis=1, norm='l1', copy=False)
```
- **Purpose**: Scale all features to [0,1] range using L1 normalization
- **Why**: Improves model performance by ensuring features are on comparable scales

---

### 4. **Train/Test Split**

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```
- **Purpose**: Divide data into 70% training and 30% testing subsets
- **random_state=42**: Ensures reproducible results across runs
- **Why**: Training set builds the model; test set evaluates its generalization ability

---

### 5. **Build Decision Tree Regressor**

```python
dt_reg = DecisionTreeRegressor(
    criterion='squared_error',  # Minimize sum of squared errors
    max_depth=8,                # Limit tree depth to prevent overfitting
    random_state=35
)
dt_reg.fit(X_train, y_train)
```

#### How It Works
1. **criterion='squared_error'**: Splits nodes to minimize prediction variance (error)
2. **max_depth=8**: Tree grows up to 8 levels deep, controlling model complexity
3. **fit()**: Recursively partitions feature space by creating decision rules (e.g., "if fare > $12, go left")

---

### 6. **Model Evaluation**

```python
y_pred = dt_reg.predict(X_test)
mse_score = mean_squared_error(y_test, y_pred)
r2_score = dt_reg.score(X_test, y_test)
```

#### Metrics
- **MSE (Mean Squared Error)**: Average squared difference between predicted and actual tips
  - Lower is better
  - Units: (dollars)²
  
- **R² Score**: Coefficient of determination (0 to 1)
  - 1.0 = perfect predictions
  - 0.0 = model performs as well as always predicting the mean
  - Negative = model worse than baseline

---

## Practice Questions & Solutions

### Q1: Effect of max_depth=12
**Question**: What happens if we increase max_depth to 12?

**Answer**: 
- Tree becomes more complex and may **overfit** to training data
- Test MSE typically **increases** (worse performance)
- R² typically **decreases** (may become negative)

### Q2: Top 3 Features
**Code**:
```python
corr_vals = raw_data.corr()['tip_amount'].drop('tip_amount')
top3 = corr_vals.abs().sort_values(ascending=False).head(3)
```
**Result**: `fare_amount`, `trip_distance`, and `tolls_amount` have strongest correlation with tip_amount (makes intuitive sense!)

### Q3: Dropping Low-Correlation Features
**Code**:
```python
raw_data_dropped = raw_data.drop(['payment_type', 'VendorID', 'store_and_fwd_flag', 'improvement_surcharge'], axis=1)
# Retrain with reduced feature set
```
**Finding**: MSE and R² remain nearly unchanged → these features don't improve predictions

### Q4: Effect of max_depth=4
**Question**: What happens if we decrease max_depth to 4?

**Answer**:
- Simpler tree = less overfitting
- Test MSE typically **decreases** (better performance)
- R² typically **increases**
- Suggests `max_depth=4` is better suited for this dataset than the original `max_depth=8`

---

## Key Takeaways

1. **Feature Selection Matters**: Dropping uncorrelated features doesn't hurt performance and reduces complexity
2. **Hyperparameter Tuning**: `max_depth` controls model complexity; find the sweet spot to avoid overfitting
3. **Evaluation Metrics**: Use both MSE and R² to assess model quality
4. **Data Preprocessing**: Normalization ensures fair feature weighting in tree splits

---

## Running the Notebook

1. Execute cells sequentially from top to bottom
2. Each practice question (Q1-Q4) has a code cell to run independently
3. Modify hyperparameters and re-run to observe effects on model performance

---

## References
- [Scikit-Learn DecisionTreeRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)
- [NYC TLC Trip Record Data](https://www1.nyc.gov/site/tlc/about/tlc-trip-record-data.page)
- Author: Abhishek Gagneja
