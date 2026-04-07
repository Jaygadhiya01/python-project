# Practical Definition:

# A dataset contains information related to student academic performance. The dataset includes the following columns:

# Student_ID
# Student_Name
# Gender
# Attendance
# Study_Hours
# Assignment_Marks
# Internal_Marks
# Final_Marks
# Exam_Date

# The dataset is not in a proper condition and contains several common data problems:

# Some numerical columns contain missing values
# Some rows are duplicated
# Some columns such as Attendance or Study_Hours may be stored as text instead of numeric values
# The Exam_Date column is stored as string instead of proper date format

# You are required to perform complete data cleaning on this dataset.

# Tasks to Perform:
# Load the dataset into Python using Pandas.
# Display the first few rows and inspect the overall structure of the dataset.
# Identify all missing values present in the dataset.
# Count missing values column-wise.
# Handle missing values using suitable techniques such as:
# Mean
# Median
# Mode
# Row removal (if necessary)
# Detect and remove duplicate records from the dataset.
# Check the data types of all columns.
# Convert incorrect data types into appropriate formats:
# Numeric columns → integer or float
# Date columns → datetime
# Display the cleaned dataset after applying all transformations.
# Compare the dataset before and after cleaning.
# Write a short conclusion explaining:
# What issues were found in the dataset
# Which cleaning techniques were applied
# Why data cleaning is important before data analysis or machine learning
# Expected Learning Outcome:

# After completing this practical, the dataset should become clean, consistent, and analysis-ready. This practical helps in understanding how to handle real-world raw data before using it in statistical analysis, visualization, or machine learning models.

# Real-World Relevance:

# This type of data cleaning is commonly used in:

# Student performance analysis
# Educational data mining
# Result prediction systems
# Attendance-based performance modeling


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

# Create raw dataset with issues
data = {
    "Student_ID": [101, 102, 103, 104, 105, 105],
    "Student_Name": ["Amit", "Neha", "Raj", "Priya", "John", "John"],
    "Gender": ["M", "F", "M", "F", "M", "M"],
    "Attendance": ["85", "90", None, "eighty", "75", "75"],
    "Study_Hours": ["2", "3.5", "four", None, "5", "5"],
    "Assignment_Marks": [80, None, 70, 85, 90, 90],
    "Internal_Marks": [75, 88, None, 80, 85, 85],
    "Final_Marks": [78, 92, 68, None, 88, 88],
    "Exam_Date": ["2024-01-10", "2024-01-12", "2024/01/15", "15-01-2024", None, None]
}

df = pd.DataFrame(data)

print("Raw Dataset:")
print(df)



print("\nfirst 5 Rows:")
print(df.head())

print("\ndataset Info:")
print(df.info())


print("\nmissing values:")
print(df.isnull().sum())


df["Attendance"] = pd.to_numeric(df["Attendance"], errors='coerce')
df["Study_Hours"] = pd.to_numeric(df["Study_Hours"], errors='coerce')
df["Exam_Date"] = pd.to_datetime(df["Exam_Date"], errors='coerce')
df["Exam_Date"] = df["Exam_Date"].ffill()

df["Attendance"] = df["Attendance"].fillna(df["Attendance"].mean())
df["Study_Hours"] = df["Study_Hours"].fillna(df["Study_Hours"].median())

df["Assignment_Marks"] = df["Assignment_Marks"].fillna(df["Assignment_Marks"].mean())
df["Internal_Marks"] = df["Internal_Marks"].fillna(df["Internal_Marks"].mean())
df["Final_Marks"] = df["Final_Marks"].fillna(df["Final_Marks"].mean())

df["Gender"] = df["Gender"].fillna(df["Gender"].mode()[0])



print("\nmissing values:")
print(df.isnull().sum())



print("\nBefore Removing Duplicates:", df.shape)
df.drop_duplicates(inplace=True)
print("After Removing Duplicates:", df.shape)






df = pd.get_dummies(df, columns=["Gender"], drop_first=True)



print("\nCleaned Dataset:")
print(df)


X = df[["Attendance", "Study_Hours", "Assignment_Marks", "Internal_Marks", "Gender_M"]]
y = df["Final_Marks"]



X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.4, random_state=42 )


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


model = LinearRegression()
model.fit(X_train, y_train)



y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))



# # Example new student
# new_data = [[85, 4, 80, 78, 1]]  # Gender_M = 1

# new_data = scaler.transform(new_data)

# prediction = model.predict(new_data)

# print("Predicted Final Marks:", prediction[0])