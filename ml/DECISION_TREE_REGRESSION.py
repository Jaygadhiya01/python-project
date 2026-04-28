# ==========================================
# DECISION TREE REGRESSION
# REAL-LIFE EXAMPLE: Experience vs Salary
# ==========================================

# Step 1: Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeRegressor

# Step 2: Create dataset
data = {
    "Experience": [1,2,3,4,5,6,7,8,9,10],
    "Salary":     [25000,28000,32000,39000,48000,60000,75000,93000,114000,138000]
}

df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["Experience"]]
y = df["Salary"]

# Step 4: Create model
model = DecisionTreeRegressor()

# Step 5: Train model
model.fit(X, y)

# Step 6: Predictions
y_pred = model.predict(X)

print("\n===== PREDICTIONS =====")
print(y_pred)

# Step 7: Predict new value
new_exp = pd.DataFrame({"Experience": [5.5]})
new_salary = model.predict(new_exp)

print("\n===== NEW PREDICTION =====")
print("Predicted Salary:", new_salary[0])

# Step 8: Visualization
plt.scatter(X, y, label="Actual Data")

# Smooth X for step graph
X_range = np.arange(min(X.values), max(X.values), 0.01).reshape(-1,1)
y_range_pred = model.predict(X_range)

plt.plot(X_range, y_range_pred, label="Decision Tree Prediction")

plt.xlabel("Experience")
plt.ylabel("Salary")
plt.title("Decision Tree Regression")
plt.legend()
plt.show()