import pandas as pd

df = pd.read_csv("stock_details_5_years.csv")
print(df.head())


# df["Date"]=pd.to_datetime(df["Date"])
# df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
df['Date'] = pd.to_datetime(df['Date'], utc=True).dt.tz_convert(None)




print(df.info())
print(df.describe())



companies = df['Company'].unique()
print(companies)


import matplotlib.pyplot as plt

aapl = df[df['Company'] == 'AAPL']

plt.figure(figsize=(10,5))
plt.plot(aapl['Date'], aapl['Close'])
plt.title("AAPL Closing Price Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.show()



aapl['MA_50'] = aapl['Close'].rolling(50).mean()
aapl['MA_200'] = aapl['Close'].rolling(200).mean()


plt.figure(figsize=(10,5))
plt.plot(aapl['Date'], aapl['Close'], label='Close')
plt.plot(aapl['Date'], aapl['MA_50'], label='50 Day MA')
plt.plot(aapl['Date'], aapl['MA_200'], label='200 Day MA')
plt.legend()
plt.show()



aapl['Daily_Return'] = aapl['Close'].pct_change()

plt.figure(figsize=(10,4))
plt.plot(aapl['Date'], aapl['Daily_Return'])
plt.title("AAPL Daily Returns")
plt.show()



plt.figure(figsize=(10,4))
plt.bar(aapl['Date'], aapl['Volume'])
plt.title("AAPL Trading Volume")
plt.show()



plt.figure(figsize=(12,6))

for comp in companies:
    temp = df[df['Company'] == comp]
    plt.plot(temp['Date'], temp['Close'], label=comp)

plt.legend()
plt.title("Stock Price Comparison")
plt.show()



returns = df.groupby('Company')['Close'].pct_change()

df['Return'] = returns

avg_return = df.groupby('Company')['Return'].mean()
avg_return




volatility = df.groupby('Company')['Return'].std()
volatility


import seaborn as sns

price_data = df.pivot(index='Date', columns='Company', values='Close')

sns.heatmap(price_data.corr(), annot=True, cmap='coolwarm')
plt.title("Stock Price Correlation")
plt.show()


