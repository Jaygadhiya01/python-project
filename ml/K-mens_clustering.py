# ==========================================
# K-MEANS CLUSTERING
# REAL-LIFE EXAMPLE: Customer Segmentation
# ==========================================

# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

# Step 2: Create dataset
data = {
    "Income": [15,16,17,18,19,50,52,54,56,58],
    "Spending": [39,40,42,43,45,60,62,64,66,68]
}

df = pd.DataFrame(data)

print("===== DATASET =====")
print(df)

# Step 3: Define X (no y!)
X = df[["Income", "Spending"]]

# Step 4: Create model (K=2)
model = KMeans(n_clusters=2)

# Step 5: Train model
model.fit(X)

# Step 6: Get cluster labels
labels = model.labels_

print("\n===== CLUSTER LABELS =====")
print(labels)

# Step 7: Add labels to dataset
df["Cluster"] = labels

print("\n===== FINAL DATA =====")
print(df)

# Step 8: Plot clusters
plt.scatter(df["Income"], df["Spending"], c=df["Cluster"])

plt.xlabel("Income")
plt.ylabel("Spending Score")
plt.title("K-Means Clustering")
plt.show()