# ==========================================
# KNN CLASSIFICATION
# REAL-LIFE EXAMPLE: Student Pass/Fail
# ==========================================

# Step 1: Import libraries
import pandas as pd

from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Step 2: Create dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8],
    "Attendance":  [50,55,60,65,70,75,80,85],
    "Result":      [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

# Step 4: Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Create model
model = KNeighborsClassifier(n_neighbors=3)

# Step 6: Train model
model.fit(X_scaled, y)

# Step 7: Predictions
y_pred = model.predict(X_scaled)

print("\n===== PREDICTIONS =====")
print(y_pred)

# Step 8: New prediction
new_student = pd.DataFrame({
    "Study_Hours": [5],
    "Attendance": [72]
})

new_scaled = scaler.transform(new_student)

new_pred = model.predict(new_scaled)

print("\n===== NEW PREDICTION =====")
print("Predicted Class:", new_pred[0])