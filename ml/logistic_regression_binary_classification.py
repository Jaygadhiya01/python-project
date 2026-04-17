# ==========================================
# LOGISTIC REGRESSION (BINARY CLASSIFICATION)
# REAL-LIFE EXAMPLE: Student Pass/Fail
# ==========================================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression

# Step 2: Create dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8],
    "Result":      [0,0,0,1,1,1,1,1]   # 0 = Fail, 1 = Pass
}

df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["Study_Hours"]]
y = df["Result"]

print("\n===== INPUT (X) =====")
print(X)

print("\n===== OUTPUT (y) =====")
print(y)

# Step 4: Create model
model = LogisticRegression()

# Step 5: Train model
model.fit(X, y)

# Step 6: Predict class
y_pred = model.predict(X)

print("\n===== PREDICTED CLASSES =====")
print(y_pred)

# Step 7: Predict probabilities
y_prob = model.predict_proba(X)

print("\n===== PREDICTED PROBABILITIES =====")
for i in range(len(X)):
    print(f"Study Hours = {X.iloc[i,0]} → Probability of Pass = {round(y_prob[i][1], 3)}")

# Step 8: New prediction
new_data = pd.DataFrame({"Study_Hours": [3.5]})
new_pred = model.predict(new_data)
new_prob = model.predict_proba(new_data)

print("\n===== NEW PREDICTION =====")
print("Predicted Class:", new_pred[0])
print("Probability of Pass:", round(new_prob[0][1], 3))

# Step 9: Visualization (Decision Boundary)
plt.scatter(X, y, color="blue", label="Data Points")

# Smooth curve
X_range = np.linspace(X.min(), X.max(), 100).reshape(-1,1)
y_range_prob = model.predict_proba(X_range)[:,1]

plt.plot(X_range, y_range_prob, color="red", label="Sigmoid Curve")

plt.xlabel("Study Hours")
plt.ylabel("Probability of Pass")
plt.title("Logistic Regression Curve")
plt.legend()
plt.show()