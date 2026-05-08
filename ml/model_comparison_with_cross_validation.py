
# ==========================================
# MODEL COMPARISON USING CROSS VALIDATION
# REAL-LIFE EXAMPLE: Student Pass/Fail
# ==========================================

# Step 1: Import libraries
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import cross_val_score

# Step 2: Create dataset
# data = {
#     "Study_Hours": [-10,2,3,4,5,6,7,8,9,100],
#     "Result":      [0,0,0,0,1,1,1,1,1,1]
# }

# Step 2: Create a larger, more realistic dataset
data = {
    "Study_Hours": [1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 6.5, 7, 7.5, 8, 8.5, 9, 9.5, 10, 11],
    "Result":      [0, 0,   0, 1,   0, 0,   1, 0,   1, 1,   0, 1,   1, 1,   0, 1,   1, 1,  1,  1]
}


df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["Study_Hours"]]
y = df["Result"]

# Step 4: Create models
log_model = LogisticRegression()
dt_model = DecisionTreeClassifier()
knn_model = KNeighborsClassifier(n_neighbors=3)

# Step 5: Apply Cross Validation

log_scores = cross_val_score(log_model, X, y, cv=5)
dt_scores = cross_val_score(dt_model, X, y, cv=5)
knn_scores = cross_val_score(knn_model, X, y, cv=5)

# Step 6: Print scores
print("\n===== LOGISTIC REGRESSION =====")
print("Scores:", log_scores)
print("Average:", round(log_scores.mean(), 3))

print("\n===== DECISION TREE =====")
print("Scores:", dt_scores)
print("Average:", round(dt_scores.mean(), 3))

print("\n===== KNN =====")
print("Scores:", knn_scores)
print("Average:", round(knn_scores.mean(), 3))