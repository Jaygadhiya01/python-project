# ==========================================
# CONFUSION MATRIX & CLASSIFICATION METRICS
# REAL-LIFE EXAMPLE: Student Pass/Fail
# ==========================================

# Step 1: Import libraries
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# Step 2: Create dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8],
    "Result":      [0,0,0,0,1,1,1,1]
}

df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["Study_Hours"]]
y = df["Result"]

# Step 4: Train model
model = LogisticRegression()
model.fit(X, y)

# Step 5: Predictions
y_pred = model.predict(X)

print("\n===== PREDICTED VALUES =====")
print(y_pred)

# Step 6: Confusion Matrix
cm = confusion_matrix(y, y_pred)

print("\n===== CONFUSION MATRIX =====")
print(cm)

# Step 7: Metrics
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print("\n===== METRICS =====")
print("Accuracy :", round(accuracy, 3))
print("Precision:", round(precision, 3))
print("Recall   :", round(recall, 3))
print("F1 Score :", round(f1, 3))