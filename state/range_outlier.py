import numpy as np

data=[10,20,30,40,50]

print(max(data)-min(data))
print(np.var(data))

print(np.std(data))


print(np.percentile(data,25))
print(np.percentile(data,75))

print("max",np.max(data))
print("min",np.min(data))



q1=np.percentile(data,25)
q3=np.percentile(data,75)

iqr=q3-q1

print(f"lower : {q1-1.5*iqr}")
print(f"higer : {q3+1.5*iqr}")
