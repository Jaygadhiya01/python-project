import numpy as np
from statsmodels.stats.weightstats import ztest

# Sample battery life in hours
battery_life = [10.5, 9.8, 10.2, 9.7, 10.3, 10.1, 9.9, 10.0, 10.4, 10.2, 10.3, 9.8, 10.1, 10.0, 10.2]

# Population mean
population_mean = 10

# Perform one-sample z-test
z_stat, p_value = ztest(battery_life, value=population_mean)

print("Z Statistic:", z_stat) #std 
print("P Value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis: Sample differs from population")
else:
    print("Fail to Reject Null Hypothesis: No significant difference")
