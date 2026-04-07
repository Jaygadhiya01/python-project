import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler

data = {
    "patient_id": [1, 2, 3, 4, 5],
    "age": ["25", "30", "35", "40", "45"],  # string
    "weight": [60, 70, 80, 90, 100],
    "height": [160, 165, 170, 175, 180],
    "blood_sugar": ["90", "110", "140", "160", "200"]  # string
}

df = pd.DataFrame(data)
print(df)



# Convert to numeric
df["age"] = pd.to_numeric(df["age"])
df["blood_sugar"] = pd.to_numeric(df["blood_sugar"])

print(df.dtypes)



scaler = StandardScaler()

# Select columns to scale
cols = ["age", "weight", "height", "blood_sugar"]

df_standardized = df.copy()
df_standardized[cols] = scaler.fit_transform(df[cols])

print("Standardized Data:\n", df_standardized)




minmaxscaler = MinMaxScaler()

df_normalized = df.copy()
df_normalized[cols] = minmaxscaler.fit_transform(df[cols])

print("Normalized Data:\n", df_normalized)
