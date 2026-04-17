# ==========================================
# LOGISTIC REGRESSION WITH MULTIPLE INPUTS
# REAL-LIFE EXAMPLE: Student Pass/Fail
# ==========================================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

# Step 2: Create dataset
data = {
    "Study_Hours":    [1,2,3,4,5,6,7,8],
    "Attendance":     [50,55,60,65,70,75,80,85],
    "Previous_Marks": [40,45,50,55,60,65,70,75],
    "Result":         [0,0,0,0,1,1,1,1]   # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["Study_Hours", "Attendance", "Previous_Marks"]]
y = df["Result"]

print("\n===== INPUT (X) =====")
print(X)

print("\n===== OUTPUT (y) =====")
print(y)

# Step 4: Create model
model = LogisticRegression()

# Step 5: Train model
model.fit(X, y)

# Step 6: Predict classes
y_pred = model.predict(X)

print("\n===== PREDICTED CLASSES =====")
print(y_pred)

# Step 7: Predict probabilities
y_prob = model.predict_proba(X)

print("\n===== PREDICTED PROBABILITIES =====")
for i in range(len(X)):
    print(f"Data {i+1} → Probability of Pass = {round(y_prob[i][1], 3)}")

# Step 8: Check coefficients
print("\n===== MODEL COEFFICIENTS =====")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

# Step 9: New prediction
new_student = pd.DataFrame({
    "Study_Hours": [5],
    "Attendance": [72],
    "Previous_Marks": [60]
})

new_pred = model.predict(new_student)
new_prob = model.predict_proba(new_student)

print("\n===== NEW STUDENT PREDICTION =====")
print("Predicted Class:", new_pred[0])
print("Probability of Pass:", round(new_prob[0][1], 3))