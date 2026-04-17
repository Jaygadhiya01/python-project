import pandas as pd


data = {
    "study_hours":[1,2,3,4,5,6,7,8],
    "result" : ["fail","fail","fail","pass","pass","pass","pass","pass"]
}

df=pd.DataFrame(data)

print("dataset")
print(df)

x= df[["study_hours"]]
y= df["result"]

print("input ")
print(x)

print("output")
print(y)

print("check unique value ")
print(y.unique())

print("class conut")
print(y.value_counts())

num_class = len(y.unique())

if num_class == 2:
    print("\n this is binary classification problem")
else :
    print("\nthis is a multiclass classification problem")

