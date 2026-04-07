# Topic: Simple Linear Regression
# Question:

# Create a Simple Linear Regression model to predict House Rent based on House Size (in square feet).

# Use the given dataset and perform the following tasks:

# Dataset:
# House_Size (sq ft)	Rent (₹)
# 500	8000
# 600	9500
# 700	11000
# 800	12500
# 900	14000
# 1000	15500
# 1100	17000
# 1200	18500
# 1300	20000
# 1400	22000
# Tasks for Students:
# Create a DataFrame using the given dataset.
# Separate the data into:
# X (Independent Variable) = House_Size
# y (Dependent Variable) = Rent
# Apply Train-Test Split using:
# test_size = 0.2
# random_state = 42
# Create a Simple Linear Regression model.
# Train the model using training data.
# Predict the rent for test data.
# Display Actual Rent vs Predicted Rent.
# Print the Slope and Intercept.
# Print the Regression Equation in the form:
# Rent=𝑚×House_Size+𝑐
# Rent=m×House_Size+c
# Predict the rent for a new house size = 1500 sq ft.
# Plot the scatter graph and regression line.


import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# 1. Create DataFrame
data = {
    "House_Size": [500,600,700,800,900,1000,1100,1200,1300,1400],
    "Rent": [8000,9500,11000,12500,14000,15500,17000,18500,20000,22000]
}



df= pd.DataFrame(data)

print(df)



# 2. Separate X and y
x= df[["House_Size"]]
y = df["Rent"]


# 3. Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x,y , test_size= 0.2, random_state= 42)

print("x train \n",x_train)
print("x test \n",x_test)

print("y train \n",y_train)
print("y test \n",y_test)

# 4. Create model

model = LinearRegression()


model.fit(x_train,y_train)




# 6. Predict on test data


y_pred= model.predict(x_test)

print(np.round(y_pred,2))


# 7. Display Actual vs Predicted

print("actual  vs predicted : \n")
for actual,pred in zip(y_test,y_pred):
    print(f"actual : { actual} | predicted : {np.round(pred,2)} ")



# 8. Slope and Intercept

slope= model.coef_[0]
intercept= model.intercept_



print("\nevalution of model , values are: ")
print("slope",slope)
print("intercept",intercept)



# 9. Regression Equation
print("====regression evalution====\n")
print(f"evalution :  rent = {np.round(slope,2)} * house_size + {np.round(intercept,2) }")


# 10. Predict for new value (1500 sq ft)



new_size = pd.DataFrame([[1500]], columns=["House_Size"])
new_rent= model.predict(new_size)

print(f"\n predicted rent for 1500 sq ft :  {np.round(new_rent[0],2)} ")


# 11. Plot Graph

plt.scatter(x,y,color="blue",label = "actual data")
plt.plot(x, model.predict(x), color= 'red', label=" regression  line ")
plt.xlabel("house size in sq ft ")
plt.ylabel("Rent")
plt.title("house rent prediction ") 
plt.legend()
plt.show()