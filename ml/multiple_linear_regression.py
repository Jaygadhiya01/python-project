import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data={
    "House-size":[1000,1200,1500,1800,2000,2200,2500,2700,3000,3300],
    "Age":[10,8,7,6,5,4,3,3,2,1],
    "Bedroom":[2,2,3,3,4,4,4,5,5,5],
    "Price":[30,36,45,52,60,66,75,80,90,96]
}

df=pd.DataFrame(data)

print("=====DataSet====")
print(df)

x=df[["House-size","Age","Bedroom"]]
y=df["Price"]


print("======Input data=======")
print(x)
print("======Output data=======")
print(y)


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

print("===X Train===")
print(x_train)
print("===Y Train===")
print(y_train)
print("===X Test===")
print(x_test)
print("===Y Test===")
print(y_test)



import numpy as np
#create model
model=LinearRegression()

#train model
model.fit(x_train,y_train)

#prediction for y_train data
y_pred=model.predict(x_test)

for actual,predicted in zip(y_test,y_pred):
    print(f"Actual price {actual}lakh predicted price= {np.round(predicted,2)}lakh ")

# coeffiecient and intercept
print("===model coeffiecient===")
print("coeffiecient for House_size",np.round(model.coef_[0],4))
print("coeffiecient for Age",np.round(model.coef_[1],4))
print("coeffiecient for Bedroom",np.round(model.coef_[2],4))

print("Intercept",np.round(model.intercept_,4))



print("===Multiple Linear Regression Equation===")
print(f"price = {np.round(model.intercept_,2)} +" f"({np.round(model.coef_[0],4)} * House-size) +" f"({np.round(model.coef_[1],4)} * Age) +" f"({np.round(model.coef_[2],4)} *Bedroom) ")


new_house=pd.DataFrame({
    "House-size":[2400],
    "Age":[4],
    "Bedroom":[3]
})

new_prediction=model.predict(new_house)

print("===New house prediction====")
print(f"predicted price for new house = {np.round(new_prediction,2)} lakh")
