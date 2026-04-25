# ==========================================
# IMBALANCED DATASET HANDLING
# REAL-LIFE EXAMPLE: Pass/Fail Prediction
# ==========================================

# Step 1: Import libraries
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Step 2: Create imbalanced dataset
data = {
    "Study_Hours": [1,2,3,4,5,6,7,8,9,10],
    "Result":      [1,1,1,1,1,1,1,1,1,0]   # 9 Pass, 1 Fail
}

df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["Study_Hours"]]
y = df["Result"]

# ==========================================
# NORMAL MODEL
# ==========================================

model = LogisticRegression()
model.fit(X, y)

y_pred = model.predict(X)

print("\n===== NORMAL MODEL =====")
print(confusion_matrix(y, y_pred))
print(classification_report(y, y_pred))

# ==========================================
# BALANCED MODEL
# ==========================================

model_balanced = LogisticRegression(class_weight='balanced')
model_balanced.fit(X, y)

y_pred_bal = model_balanced.predict(X)

print("\n===== BALANCED MODEL =====")
print(confusion_matrix(y, y_pred_bal))
print(classification_report(y, y_pred_bal))