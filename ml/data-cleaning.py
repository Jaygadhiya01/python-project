import numpy as np 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor



# -------------------------------
# 1. Load Data
# -------------------------------
df = pd.read_csv("Exam_Score_Prediction.csv")

print("First Rows:\n", df.head())
print("\nShape:", df.shape)
print("\nInfo:")
print(df.info())

# -------------------------------
# 2. Handle Missing Values
# -------------------------------
print("\nMissing Values:\n", df.isnull().sum())

# Instead of dropping all rows, fill numeric with mean
for col in df.select_dtypes(include=np.number).columns:
    df[col] = df[col].fillna(df[col].mean())

# Fill categorical with mode
for col in df.select_dtypes(include=['object', 'string']).columns:
    df[col] = df[col].fillna(df[col].mode()[0])
# -------------------------------
# 3. Remove Duplicates
# -------------------------------
df = df.drop_duplicates()

# -------------------------------
# 4. Convert Data Types
# -------------------------------
df["internet_access"] = df["internet_access"].map({"yes":1, "no":0})

# -------------------------------
# 5. Encode Categorical Data
# -------------------------------
label_cols = ["gender", "course", "sleep_quality", "study_method", "facility_rating", "exam_difficulty"]

le = LabelEncoder()

for col in label_cols:
    df[col] = le.fit_transform(df[col])

# -------------------------------
# 6. Outlier Detection (Optional)
# -------------------------------
sns.boxplot(x=df["study_hours"])
plt.title("Study Hours Outliers")
plt.show()

# -------------------------------
# 7. Feature Selection
# -------------------------------
X = df.drop(columns=["exam_score", "student_id"])
y = df["exam_score"]

# -------------------------------
# 8. Train-Test Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 9. Feature Scaling
# -------------------------------
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# -------------------------------
# 10. Model Training
# -------------------------------
model = RandomForestRegressor(n_estimators=100, random_state=42)

model.fit(X_train, y_train)

# -------------------------------
# 11. Prediction
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 12. Evaluation
# -------------------------------
print("\nMAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# -------------------------------
# 13. Feature Importance
# -------------------------------
importances = model.feature_importances_
feature_names = X.columns

feat_imp = pd.Series(importances, index=feature_names)
feat_imp.sort_values().plot(kind='barh')

plt.title("Feature Importance")
plt.show()




# -------------------------------
# 14. Predict Exam Score for New Data
# -------------------------------

# Example new students
new_students = pd.DataFrame({
    "age": [18, 21, 22],
    "gender": ["female", "male", "other"],
    "course": ["diploma", "b.sc", "bca"],
    "study_hours": [3.5, 6.0, 2.0],
    "class_attendance": [85.0, 92.0, 70.0],
    "internet_access": ["yes", "yes", "no"],
    "sleep_hours": [7.5, 8.0, 6.0],
    "sleep_quality": ["good", "poor", "average"],
    "study_method": ["coaching", "online videos", "self study"],
    "facility_rating": ["high", "medium", "low"],
    "exam_difficulty": ["medium", "hard", "moderate"]
})

# Encode new data same as training
# internet_access
new_students["internet_access"] = new_students["internet_access"].map({"yes":1, "no":0})

# Label encode categorical columns (same as before)
for col in label_cols:
    new_students[col] = le.fit_transform(new_students[col])

# Feature scaling (if you used scaler)
X_new = scaler.transform(new_students)

# Predict
predicted_scores = model.predict(X_new)

# Show predictions
new_students["predicted_exam_score"] = predicted_scores
print("\nPredicted Exam Scores for New Students:")
print(new_students)