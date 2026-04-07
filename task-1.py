
import pandas as pd

df =pd.read_csv("customer_purchase_data.csv")

print(df.head)


print(df.shape)


df["Purchase_Date"]=pd.to_datetime(df["Purchase_Date"])

print("df ", df)


import numpy as np



avg_total=np.mean(df["Total_Amount"])


print("avg total amount ",avg_total)

avg_unit_price=np.mean(df["Units_Purchased"])


print("avg Units_Purchased", avg_unit_price)





max_amount= np.max(df["Total_Amount"])

mim_amount=np.min(df["Total_Amount"])

print("max amount ",max_amount)
print("min amonnt",mim_amount)



salse_by_group=df.groupby("Gender")["Total_Amount"].sum()

print("salse by gender:",salse_by_group)



salse_by_Region=df.groupby("Region")["Total_Amount"].sum()

print("salse by Region:",salse_by_Region)




salse_by_catagory=df.groupby("Product_Category")["Total_Amount"].sum()

print("salse by catagory :",salse_by_catagory)




df["month"]=df["Purchase_Date"].dt.month

print(df["month"])

monthly_sales=df.groupby("month")["Total_Amount"].sum()

print("monthly sales",monthly_sales)





df["week"]=df["Purchase_Date"].dt.isocalendar().week.astype(int)

print(df["week"])

week_sales=df.groupby("week")["Total_Amount"].sum()

print("week sales",week_sales)



import matplotlib.pyplot as plt
import seaborn as sns

# If not already done
df["Purchase_Date"] = pd.to_datetime(df["Purchase_Date"])



sales_by_category = df.groupby("Product_Category")["Total_Amount"].sum()

plt.figure()
sales_by_category.plot(kind="bar")
plt.title("Total Sales by Product Category")
plt.xlabel("Product Category")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()


region_count = df["Region"].value_counts()

plt.figure()
plt.pie(region_count, labels=region_count.index, autopct="%1.1f%%", startangle=90)
plt.title("Customer Distribution by Region")
plt.show()


daily_sales = df.groupby("Purchase_Date")["Total_Amount"].sum()

plt.figure()
plt.plot(daily_sales.index, daily_sales.values)
plt.title("Daily Total Sales Over Time")
plt.xlabel("Date")
plt.ylabel("Total Sales")
plt.xticks(rotation=45)
plt.show()


plt.figure()
sns.boxplot(x="Gender", y="Total_Amount", data=df)
plt.title("Total Amount Distribution by Gender")
plt.show()


numeric_df = df[["Age", "Units_Purchased", "Price_Per_Unit", "Total_Amount"]]

plt.figure()
sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

