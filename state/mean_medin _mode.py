
import numpy as np

import statistics

np.random.seed(5)
x=np.random.randint(50,100,10)
print(x)



print(np.mean(x))

sort_x=np.sort(x)

print(np.median(sort_x))


print(statistics.mode(sort_x))
