
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import mean_absolute_error, mean_squared_error 

# Step 2: Create dataset 
data = { 
"Experience": [1,2,3,4,5,6,7,8,9,10], 
"Salary":     [25000,28000,32000,39000,48000,60000,75000,93000,114000,138000] 
} 

df = pd.DataFrame(data) 
print("===== DATASET =====") 
print(df) 

# Step 3: Define X and y 
X = df[["Experience"]] 
y = df["Salary"]
# ========================================== 
# LINEAR REGRESSION 
# ========================================== 

# Step 4: Train Linear Model 
linear_model = LinearRegression() 
linear_model.fit(X, y)

# Step 5: Predictions 
y_pred_linear = linear_model.predict(X) 
# ========================================== 
# POLYNOMIAL REGRESSION 
# ========================================== 

# Step 6: Create polynomial features 
poly = PolynomialFeatures(degree=2) 
X_poly = poly.fit_transform(X) 



# Step 7: Train Polynomial Model 
poly_model = LinearRegression() 
poly_model.fit(X_poly, y)

# Step 8: Predictions 
y_pred_poly = poly_model.predict(X_poly) 

# ========================================== 
# ERROR METRICS CALCULATION 
# ========================================== 

# Step 9: Linear Regression Errors 
mae_linear = mean_absolute_error(y, y_pred_linear) 
mse_linear = mean_squared_error(y, y_pred_linear) 
rmse_linear = np.sqrt(mse_linear)
 
# Step 10: Polynomial Regression Errors 
mae_poly = mean_absolute_error(y, y_pred_poly) 
mse_poly = mean_squared_error(y, y_pred_poly) 
rmse_poly = np.sqrt(mse_poly) 
# ========================================== 
# PRINT RESULTS 
# ========================================== 
print("\n===== LINEAR REGRESSION ERRORS =====") 
print("MAE :", round(mae_linear, 2)) 
print("MSE :", round(mse_linear, 2)) 
print("RMSE:", round(rmse_linear, 2)) 
print("\n===== POLYNOMIAL REGRESSION ERRORS =====") 
print("MAE :", round(mae_poly, 2)) 
print("MSE :", round(mse_poly, 2)) 
print("RMSE:", round(rmse_poly, 2)) 
# ========================================== 
# VISUALIZATION 
# ========================================== 
plt.scatter(X, y, label="Actual Data", color="blue") 
plt.plot(X, y_pred_linear, label="Linear Regression", color="red") 
plt.plot(X, y_pred_poly, label="Polynomial Regression", color="green") 
plt.xlabel("Experience") 
plt.ylabel("Salary") 
plt.title("Model Comparison") 
plt.legend() 
plt.show() 