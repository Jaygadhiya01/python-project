# ==========================================
# HYPERPARAMETER TUNING (GRID SEARCH)
# REAL-LIFE EXAMPLE: Student Pass/Fail
# ==========================================

# Step 1: Import libraries
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

# Step 2: Create dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Result":      [0,0,0,0,1,1,1,1,1,1]
}

df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["Study_Hours"]]
y = df["Result"]

# Step 4: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Create model
model = KNeighborsClassifier()

# Step 6: Define parameter grid
param_grid = {
    "n_neighbors": [1, 3, 5, 7, 9]
}

# Step 7: Apply GridSearchCV
grid = GridSearchCV(model, param_grid, cv=5)

grid.fit(X_scaled, y)

# Step 8: Best parameters
print("\n===== BEST PARAMETERS =====")
print(grid.best_params_)

# Step 9: Best score
print("\n===== BEST SCORE =====")
print(grid.best_score_)

# Step 10: Best model
best_model = grid.best_estimator_

# Step 11: New prediction
import numpy as np

new_data = np.array([[6]])
new_scaled = scaler.transform(new_data)

prediction = best_model.predict(new_scaled)

print("\n===== NEW PREDICTION =====")
print("Predicted Class:", prediction[0])
