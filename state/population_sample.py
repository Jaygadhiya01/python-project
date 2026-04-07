import numpy as np
import pandas as pd



np.random.seed(42)
population_marks=np.random.normal(loc=70,scale=10,size=1000)
print(population_marks)

df_population=pd.DataFrame(population_marks,columns=["marks"])

print(df_population.head())


df_population_mean=df_population["marks"].mean()
df_population_var=df_population["marks"].var()

print(df_population_mean,df_population_var)



sample=df_population.sample(n=100,random_state=1)

print(sample)
sample_mean=sample["marks"].mean()
sample_var=sample["marks"].var()


print(sample_mean,sample_var)



import matplotlib.pyplot as plt

plt.hist(population_marks, bins=30, alpha=0.5, label="Population")
plt.hist(sample, bins=15, alpha=0.7, label="Sample")

plt.legend()
plt.show()