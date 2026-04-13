
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression 
from sklearn.preprocessing import PolynomialFeatures 
from sklearn.metrics import r2_score

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



r2_liner = r2_score(y,y_pred_linear)
print(f" liner r2 {r2_liner}")



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





r2_ploy = r2_score(y,y_pred_poly)
print(f" r2 ploy  {r2_ploy}")


def adjusted_r2(r2,n ,k):
    return 1-((1-r2)* (n-1) / (n-k-1))


n= len(y)

k_liner= X.shape[1]

adj_r2_liner= adjusted_r2(r2_liner,n,k_liner)
print(f" adjusted r2 liner {adj_r2_liner}")


k_poly = X_poly.shape[1]-1
adj_r2_poly= adjusted_r2(r2_ploy,n,k_poly)
print(f" adjusted r2 poly {adj_r2_poly}")

# ========================================== 
# Step 9: Linear Regression Errors 
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