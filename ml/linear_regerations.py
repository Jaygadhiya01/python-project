# ==========================================
# SIMPLE LINEAR REGRESSION PRACTICAL
# REAL-LIFE EXAMPLE: EXPERIENCE vs SALARY
# ==========================================

# Step 1: Import libraries
import pandas as pd #dataset create krne k liye
import matplotlib.pyplot as plt #graphs ke liye

from sklearn.model_selection import train_test_split # data divide kr ne ke liye
from sklearn.linear_model import LinearRegression #regression model bnane ke liye

# Step 2: Create dataset
data = {
    "Experience": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    "Salary": [25000, 30000, 35000, 40000, 50000, 55000, 60000, 70000, 75000, 85000]
}
# 1 year exp. 25000
#10 year exp. 85000



df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X and y
X = df[["Experience"]]   # 2D format  #input feature
y = df["Salary"] #target

print("\n===== INPUT (X) =====")
print(X)

print("\n===== OUTPUT / TARGET (y) =====")
print(y)

# Step 4: Apply Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#20% data for testing
#fixed random split deta hai
#har baar same output ke liye random_state=42 use hota hai
print("\n===== TRAIN-TEST SPLIT =====")
print("X_train:")
print(X_train)

print("\nX_test:")
print(X_test)

print("\ny_train:")
print(y_train)

print("\ny_test:")
print(y_test)

# Step 5: Create model
model = LinearRegression()
#create model using liner regression

# Step 6: Train model
model.fit(X_train, y_train)



# Step 7: Predict on test data
y_pred = model.predict(X_test)



print("\n===== PREDICTIONS =====")
for actual, predicted in zip(y_test, y_pred):
    print(f"Actual Salary = {actual}, Predicted Salary = {round(predicted, 2)}")

# Step 8: Show slope and intercept
print("\n===== MODEL EQUATION VALUES =====")
print("Slope (m):", model.coef_[0]) # 1 year sallary:5000 2 :7000
print("Intercept (c):", model.intercept_)#0year 


# Step 9: Plot training data + regression line
plt.scatter(X_train, y_train, label="Training Data")
plt.plot(X_train, model.predict(X_train), label="Regression Line")
plt.xlabel("Experience (Years)")
plt.ylabel("Salary")
plt.title("Simple Linear Regression: Experience vs Salary")
plt.legend()
plt.show()

