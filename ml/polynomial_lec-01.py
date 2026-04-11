# polynominal regression

# linear -> 1 column 
# multiple -> multiple column
# polynominal -> non-linear

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

data={
    "Experience":[1,2,3,4,5,6,7,8,9,10],
    "salary":[25000,27000,30000,35000,42000,52000,65000,80000,98000,120000]
}

# x
# x2 5,25
# x3 5,25,125

df=pd.DataFrame(data)

print("====DataSet====")
print(df)

# step3: define x and y

x=df[["Experience"]]
y=df["salary"]


# step4 convert to polynomial features

poly=PolynomialFeatures(degree=2)

x_poly=poly.fit_transform(x)


print("===Polynomial Features===")
print(x_poly)


# create model

model=LinearRegression()

model.fit(x_poly,y)

# predict values

y_pred=model.predict(x_poly)

print("===Actual Vs Prediction===")

for actual,predicted in zip(y,y_pred):
    print(f"Actual ={actual} ,Predicted ={np.round(predicted,2)}")

print("===Model Deatils===")
print("Intercept : ",np.round(model.intercept_,2))
print("coefficients : ",model.coef_)
# b0+b1x1    


# b0+b1x1+b2x^2

print("===Polynomial Equation===")
print(f"salary = {np.round(model.intercept_,2)} + " f"({np.round(model.coef_[1],2)} * Experience) + " f"({np.round(model.coef_[2],2)} * Experience^2)")


new_Exp=pd.DataFrame({"Experience":[11]})

new_exp_poly=poly.transform(new_Exp)

new_salary=model.predict(new_exp_poly)


print("===New prediction for the salary")
print(f"predicted salary for 11 years = {np.round(new_salary[0],2)}")