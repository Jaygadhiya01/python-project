# ==========================================
# DECISION TREE CLASSIFICATION
# REAL-LIFE EXAMPLE: Student Pass/Fail
# ==========================================

# Step 1: Import libraries
import pandas as pd

from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt


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
model = DecisionTreeClassifier()

# Step 5: Train model
model.fit(X, y)

# Step 6: Predictions
y_pred = model.predict(X)

print("\n===== PREDICTIONS =====")
print(y_pred)

# Step 7: Visualize tree
plt.figure(figsize=(8,5))
tree.plot_tree(model, feature_names=X.columns, class_names=["Fail","Pass"], filled=True)
plt.title("Decision Tree")
plt.show()

# Step 8: New prediction
new_student = pd.DataFrame({
    "Study_Hours": [4],
    "Attendance": [68]
})

new_pred = model.predict(new_student)

print("\n===== NEW PREDICTION =====")
print("Predicted Class:", new_pred[0])