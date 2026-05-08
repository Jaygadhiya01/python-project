# ==========================================
# HIERARCHICAL CLUSTERING
# ==========================================

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering

# Step 2: Dataset
data = {
    "Income": [15,16,17,18,19,50,52,54,56,58],
    "Spending": [39,40,42,43,45,60,62,64,66,68]
}

df = pd.DataFrame(data)

X = df[["Income", "Spending"]]

# Step 3: Create model
model = AgglomerativeClustering(n_clusters=2)

# Step 4: Fit and predict
labels = model.fit_predict(X)

# Step 5: Add labels
df["Cluster"] = labels

print("===== FINAL DATA =====")
print(df)

# Step 6: Plot
plt.scatter(df["Income"], df["Spending"], c=df["Cluster"])

plt.xlabel("Income")
plt.ylabel("Spending")
plt.title("Hierarchical Clustering")
plt.show()



# ==========================================
# DENDROGRAM
# ==========================================

from scipy.cluster.hierarchy import dendrogram, linkage

linked = linkage(X, method='ward')

plt.figure(figsize=(8,5))
dendrogram(linked)
plt.title("Dendrogram")
plt.show()