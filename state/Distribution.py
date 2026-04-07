# import numpy as np 
# import seaborn as sns
# import matplotlib.pyplot as plt


# args1 = [23, 24, 45, 32, 45, 66, 77, 88, 44, 33]

# # Dataset with an outlier
# args2 = [23, 24, 45, 32, 45, 66, 77, 88, 44, 33, 510]

# # Plot both histograms side by side
# plt.figure(figsize=(12, 5))

# # Original data
# plt.subplot(1, 2, 1)
# sns.histplot(args1, kde=True, color='skyblue', bins=10)
# plt.title("Original Data")

# # With outlier
# plt.subplot(1, 2, 2)
# sns.histplot(args2, kde=True, color='salmon', bins=10)
# plt.title("With Outlier")

# plt.tight_layout()
# plt.show()



import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# Your original dataset
data = np.array([23, 24, 45, 32, 45, 66, 77, 88, 44, 33])

mean = data.mean()
std = data.std()
normal_data = np.random.normal(loc=mean, scale=std, size=1000)

log_data = np.log(data[data > 0]) 
log_mean = log_data.mean()
log_std = log_data.std()
lognormal_data = np.random.lognormal(mean=log_mean, sigma=log_std, size=1000)


alpha = 2.5  
powerlaw_data = (np.random.pareto(a=alpha, size=1000) + 1) * min(data)

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(normal_data, bins=20, kde=True, color='skyblue')
plt.title("Normal Distribution")

plt.subplot(1, 3, 2)
sns.histplot(lognormal_data, bins=20, kde=True, color='salmon')
plt.title("Lognormal Distribution")

plt.subplot(1, 3, 3)
sns.histplot(powerlaw_data, bins=20, kde=True, color='lightgreen')
plt.title("Power-law Distribution")

plt.tight_layout()
plt.show()