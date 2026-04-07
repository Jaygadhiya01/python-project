import numpy as np
from sklearn.linear_model import LinearRegression,LogisticRegression

# regression model
x=np.array([1,2,3,4,5]).reshape(-1,1)
y=np.array([35,40,50,65,70])


model=LinearRegression()

model.fit(x,y)
prediction=model.predict([[6]])
print("Predicted Marks:", prediction)


# classification model

a=np.array([1,2,3,4,5]).reshape(-1,1)
b=np.array([0,0,1,1,1])
model1=LogisticRegression()


model1.fit(a,b)
prediction1=model1.predict([[2.5]])

print("predict ",prediction1)