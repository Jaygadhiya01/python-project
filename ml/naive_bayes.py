# ==========================================
# NAIVE BAYES CLASSIFICATION
# REAL-LIFE EXAMPLE: Student Pass/Fail
# ==========================================

# Step 1: Import libraries
import pandas as pd

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Step 2: Create dataset
data = {
    "Study_Hours":    [1,2,3,4,5,6,7,8],
    "Attendance":     [50,55,60,65,70,75,80,85],
    "Result":         [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["Study_Hours", "Attendance"]]
y = df["Result"]

# Step 4: Create model
model = GaussianNB()

# Step 5: Train model
model.fit(X, y)

# Step 6: Predictions
y_pred = model.predict(X)

print("\n===== PREDICTIONS =====")
print(y_pred)

# Step 7: Accuracy
accuracy = accuracy_score(y, y_pred)

print("\n===== ACCURACY =====")
print(round(accuracy, 3))

# Step 8: New prediction
new_student = pd.DataFrame({
    "Study_Hours": [5],
    "Attendance": [72]
})

new_pred = model.predict(new_student)

print("\n===== NEW PREDICTION =====")
print("Predicted Class:", new_pred[0])