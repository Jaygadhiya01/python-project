import numpy as np
import matplotlib.pyplot as plt 

population= np.random.normal(50,100,10000)


sample_mean= []


for i in range(1000):
    sample = np.random.choice(population,30)
    sample_mean.append(np.mean(sample))


plt.hist(sample_mean,bins=30)
plt.title("sampling distubution of sample mean")
plt.show()