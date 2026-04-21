# PART 1 – BASIC UNDERSTANDING
# Create a DataFrame using the dataset.
# Display the dataset.
# Separate input features (X) and target variable (y).
# Count the number of input features.
# Identify the number of unique classes in the target variable.
# Determine whether the problem is binary classification or multiclass classification.


import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import accuracy_score


data = {
    "Study_Hours":    [1,2,3,4,5,6,7,8,2,3,6,7],
    "Attendance":     [50,55,60,65,70,75,80,85,58,62,78,82],
    "Previous_Marks": [40,45,50,55,60,65,70,75,48,52,68,72],
    "Result":         [0,0,0,0,1,1,1,1,0,0,1,1]
}


df= pd.DataFrame(data)

print(df)

x = df[["Study_Hours","Attendance","Previous_Marks"]]
y = df["Result"]


print(x.shape[1])

print(y.unique())




# 🔹 PART 2 – LOGISTIC REGRESSION (MULTIPLE INPUTS)
# Train a Logistic Regression model using all input features.
# Predict the class labels for the training data.
# Display the predicted output.

model = LogisticRegression()

model.fit(x,y)


y_pred = model.predict(x)
print(y_pred)



# 🔹 PART 3 – PROBABILITY ANALYSIS
# Find the predicted probabilities for each data point.
# Identify the data point with the highest probability of passing.
# Identify the data point with the lowest probability of passing.

y_prob= model.predict_proba(x)
print(y_prob)


print("\n===== PREDICTED PROBABILITIES =====")
for i in range(len(x)):
    print(f"Study Hours = {x.iloc[i,0]} → Probability of Pass = {round(y_prob[i][1], 3)}")

pass_probs = y_prob[:, 1]

max_index = pass_probs.argmax()
min_index = pass_probs.argmin()

print("Highest Probability Index:", max_index)
print("Lowest Probability Index:", min_index)




# 🔹 PART 4 – FEATURE IMPACT
# Display the model coefficients.
# Identify which feature has the highest impact on prediction.
# Identify which feature has the lowest impact on prediction.


coeff = model.coef_[0]
print(coeff)

features = x.columns

for f,c in zip(features, coeff):
    print(f"f : {f}  :::: coeff : {c}")

max_feature = features[np.argmax(coeff)]
min_feature = features[np.argmin(coeff)]

print("Highest impact:", max_feature)
print("Lowest impact:", min_feature)



# 🔹 PART 5 – NEW DATA PREDICTION
# Predict the result for the following student:
# Study Hours = 4
# Attendance = 68
# Previous Marks = 58
# Display the predicted class.
# Display the predicted probability.

new_student = [[4,68,58]]

new_pred = model.predict(new_student)
new_prob= model.predict_proba(new_student)

print("Predicted Class:", new_pred)
print("Probability of Passing:", new_prob)


# 🔹 PART 6 – POLYNOMIAL LOGISTIC REGRESSION
# Apply Polynomial Features with degree = 2.
# Transform the input data.
# Train a new Logistic Regression model using transformed data.
# Predict the class labels again.

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(x)

model_poly = LogisticRegression()
model_poly.fit(X_poly, y)

y_pred_poly = model_poly.predict(X_poly)
print(y_pred_poly)

# 🔹 PART 7 – MODEL COMPARISON
# Compare predictions from both models.
# Identify which model performs better.
# Explain the reason for performance difference.


acc_normal = accuracy_score(y, y_pred)
acc_poly = accuracy_score(y, y_pred_poly)

print("Normal Accuracy:", acc_normal)
print("Polynomial Accuracy:", acc_poly)



# 🔹 PART 8 – THINKING QUESTIONS
# What happens if one feature is removed from the dataset?
# What happens if polynomial degree is increased to 3?
# Is polynomial model always better than normal logistic regression?
# Why do we use probability in classification?
# What is the role of multiple inputs in prediction?



# 1. What if one feature is removed?
# Model loses information
# Accuracy may drop
# Example: removing Study_Hours weakens prediction

# 2. If degree = 3?
# Model becomes more complex
# Risk of overfitting increases

# 3. Is polynomial always better?
#  No
#  Only better when data is non-linear
#  Worse when data is simple (like this)

# 4. Why use probability?
# Gives confidence level
# Helps in decision-making (e.g., threshold tuning)

# 5. Role of multiple inputs?
# More features → better understanding of data
# Improves prediction accuracy
# Captures real-world complexity