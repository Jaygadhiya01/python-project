#study hours vs marks prediction

# x:study hours
# y:marks

# simple regression is supervised learning algorithm used to find linear relation between input and output

# actual value : available in dataset /real value 
# predicted value : predicted by model /answer of model

# Error/residual value = diffrence between actual value and predicted value
# error=actual value-predicted value

# actual=80
# prediction=76.4

# error=3.6
#model 3.6 marks se miss hai

# regression equation
# y=mx+c
# m=8
# x=5 (study hours)
# c=20
# y=8*5+20
# y=40+20 =60 (marks)


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection  import train_test_split
from sklearn.linear_model import LinearRegression

data={
    "study_hours":[1,2,3,4,5,6,7,8,9,10],
    "marks":[35,40,50,55,60,70,80,85,90,95,]
}

df=pd.DataFrame(data)

print("===========Dataset==========");
print(df)

x=df[["study_hours"]]
y=df[["marks"]]

print("====Input=======")
print(x)
print("====Output=======")
print(y)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


print("=====Train-Test split======")
print("X-train")
print(x_train)

print("X-Test")
print(x_test)

print("Y-Train")
print(y_train)

print("Y-Test")
print(y_test)






model=LinearRegression()

model.fit(x_train,y_train)

y_pred=model.predict(x_test)


print("=== Actual vs Predicted data===")

for actual,predicted in zip(y_test,y_pred):
    print(f"Actual marks ={actual} , predicted marks= {np.round(predicted,2)}")


slope=model.coef_[0]
intercept=model.intercept_

print("evalution of model , values are: ")
print("slope",slope)
print("intercept",intercept)




print("====regression evalution====")
print(f"marks = {np.round(slope,2)}* study_hours + {np.round(intercept,2)}")

# y=mx+c

# c=agr 0 hours , then marks what?
# m=diffrence

# y=6.89*study_hours+28.32


print("=== Error (Actual - Predicted) ===")

for actual, predicted in zip(y_test.values, y_pred):
    error = actual[0] - predicted[0]
    print(f"Actual = {actual[0]}, Predicted = {np.round(predicted[0],2)}, Error = {np.round(error,2)}")
    

new_hours = pd.DataFrame({
    "study_hours": [2, 4, 6, 8, 10]
})

predicted_marks=model.predict(new_hours)

