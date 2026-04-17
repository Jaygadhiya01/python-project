# ==========================================
# LOGISTIC REGRESSION WITH POLYNOMIAL INPUTS
# REAL-LIFE EXAMPLE: Non-Linear Classification
# ==========================================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import PolynomialFeatures

# Step 2: Create dataset (Non-linear pattern)
data = {
    "X1": [1,2,3,4,5,6,7,8],
    "X2": [8,7,6,5,4,3,2,1],
    "Class": [0,0,0,1,1,1,0,0]
}

df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["X1", "X2"]]
y = df["Class"]

# ==========================================
# LINEAR LOGISTIC REGRESSION
# ==========================================

# Step 4: Train linear model
linear_model = LogisticRegression()
linear_model.fit(X, y)

# Step 5: Predictions
y_pred_linear = linear_model.predict(X)

print("\n===== LINEAR MODEL PREDICTIONS =====")
print(y_pred_linear)

# ==========================================
# POLYNOMIAL LOGISTIC REGRESSION
# ==========================================

# Step 6: Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

print("\n===== POLYNOMIAL FEATURES =====")
print(X_poly)

# Step 7: Train model
poly_model = LogisticRegression()
poly_model.fit(X_poly, y)

# Step 8: Predictions
y_pred_poly = poly_model.predict(X_poly)

print("\n===== POLYNOMIAL MODEL PREDICTIONS =====")
print(y_pred_poly)

# Step 9: New prediction
new_data = pd.DataFrame({"X1": [4], "X2": [4]})
new_data_poly = poly.transform(new_data)

new_pred = poly_model.predict(new_data_poly)
new_prob = poly_model.predict_proba(new_data_poly)

print("\n===== NEW PREDICTION =====")
print("Predicted Class:", new_pred[0])
print("Probability:", round(new_prob[0][1], 3))