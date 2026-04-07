import numpy as np
from scipy import stats

marks = [65,70,72,68,75,80,60,62,78,74,69,71]

population_mean = 70


# One-sample t-test
t_stat, p_value = stats.ttest_1samp(marks, population_mean)

print("T Statistic:", t_stat)
print("P Value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis: Sample mean differs from population mean")
else:
    print("Fail to Reject Null Hypothesis: No significant difference")