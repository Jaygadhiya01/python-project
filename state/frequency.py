# import matplotlib as plt

# restaurants = ["Chinese", "Punjabi", "Gujarati", "South Indian", "Italian"]

# order_numbers = [50, 75, 60, 40, 65]

# plt.figure()
# plt.bar(restaurants,order_numbers)
# plt.xlabel("Restaurant Type")
# plt.ylabel("Number of Orders")
# plt.title("Restaurant Orders Bar Chart")

# plt.show()

import pandas as pd
import numpy as np

marks=[10,20,20,40,30,40,60,70,30]

marks1=pd.DataFrame(marks,columns=["student marks"])
print(marks1)



frequency=marks1.value_counts()
print(frequency)


cumalative_shorted=frequency.sort_index()
print(cumalative_shorted)

cumalative=cumalative_shorted.cumsum()
print(cumalative)





marks=np.random.randint(10,100,50)
print(marks)


marks_df=pd.DataFrame(marks)
print(marks_df)



np.random.seed(0)
x=np.random.randint(10,100,50)
print(x)


df1=pd.DataFrame(x,columns=["marks"])
print(df1)


frequency=pd.cut(df1["marks"],bins=5)

frequency_fc=frequency.value_counts()
print(frequency_fc)




np.random.seed(0)
x=np.random.randint(10,100,50)
print(x)


df1=pd.DataFrame(x,columns=["marks"])
print(df1)


frequency1=pd.cut(df1["marks"],bins=[10,30,50,70,90,110])

frequency_f1=frequency1.value_counts()
print(frequency_f1)



frequency_f2=frequency_f1.sort_index()
frequency_f2=frequency_f2.cumsum()
print(frequency_f2)
