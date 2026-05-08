import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans


from sklearn.metrics import silhouette_score

data = {
    "Income": [15,16,17,18,19,50,52,54,56,58],
    "Spending": [39,40,42,43,45,60,62,64,66,68]
}

df = pd.DataFrame(data)

X = df[["Income", "Spending"]]

# Step 3: Try multiple K values
inertia_values = []

for k in range(2, 6):
    model = KMeans(n_clusters=k)
    labels = model.fit_predict(X)
    
    score = silhouette_score(X, labels)
    
    print(f"K = {k}, Silhouette Score = {round(score,3)}")