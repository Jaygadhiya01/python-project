# ==========================================
# ELBOW METHOD
# ==========================================

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Step 2: Dataset
data = {
    "Income": [15,16,17,18,19,50,52,54,56,58],
    "Spending": [39,40,42,43,45,60,62,64,66,68]
}

df = pd.DataFrame(data)

X = df[["Income", "Spending"]]

# Step 3: Try multiple K values
inertia_values = []

for k in range(1, 6):
    model = KMeans(n_clusters=k)
    model.fit(X)
    inertia_values.append(model.inertia_)

# Step 4: Plot graph
plt.plot(range(1,6), inertia_values, marker='o')

plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()