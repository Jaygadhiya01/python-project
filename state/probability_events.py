import numpy as np

roll=np.random.randint(1,7,1000)

roll_count=sum(roll%2==0)

print(roll_count)

print(f"probability_roll : {roll_count/1000}")





coin=np.random.choice(["h","t"],1000)

coin_count=sum(coin == "h")

print(coin_count)

print(f"probability_coin head : {coin_count/1000}")




count=0
for i in range(1000):
    if coin[i] == "h" and roll[i]%2 == 0 :
        count=count + 1


print(f"total count of head with even number {count} \nprobability of head with even number {count/1000}")





deck =["Red queen","black queen","Red queen","black queen"]


# simulate many draws
deck_choice = np.random.choice(deck)

# count red queens
count_red = np.sum(deck_choice == "Red queen")

probability = count_red / 4

print(f"Probability of getting Red Queen: {probability}")






import matplotlib.pyplot as plt

data = np.random.binomial(n=10,p=0.5,size=1000)

plt.hist(data)
plt.title(" binomial graph")
plt.show()





data = np.random.randint(60,100,1000)

plt.hist(data)
plt.title(" binomial graph")
plt.show()
