# import numpy as np
# import matplotlib.pyplot as plt

# # Discrete random variable: 6-sided die
# X = np.array([1, 2, 3, 4, 5, 6])
# pmf = np.ones(6)/6  # Each outcome has probability 1/6
# cdf = np.cumsum(pmf)  # Cumulative sum for CDF

# # Print PMF and CDF
# for x_val, p_val, c_val in zip(X, pmf, cdf):
#     print(f"X = {x_val}, PMF = {p_val:.2f}, CDF = {c_val:.2f}")

# # Plot PMF
# plt.subplot(1,2,1)
# plt.bar(X, pmf, color='skyblue')
# plt.title('PMF of Die')
# plt.xlabel('X')
# plt.ylabel('P(X=x)')

# # Plot CDF
# plt.subplot(1,2,2)
# plt.step(X, cdf, where='post', color='orange')
# plt.title('CDF of Die')
# plt.xlabel('X')
# plt.ylabel('F(X≤x)')

# plt.tight_layout()
# plt.show()



from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# Continuous random variable: Normal distribution
mean = 0
std = 1
x = np.linspace(-4, 4, 1000)  # Range for X

pdf = norm.pdf(x, loc=mean, scale=std)  # PDF
cdf = norm.cdf(x, loc=mean, scale=std)  # CDF

# Plot PDF
plt.subplot(1,2,1)
plt.plot(x, pdf, color='blue')
plt.title('PDF of Normal Distribution')
plt.xlabel('X')
plt.ylabel('f(X)')

# Plot CDF
plt.subplot(1,2,2)
plt.plot(x, cdf, color='red')
plt.title('CDF of Normal Distribution')
plt.xlabel('X')
plt.ylabel('F(X≤x)')

plt.tight_layout()
plt.show()

