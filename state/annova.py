from scipy import stats

# Marks for three departments
dept1 = [65,70,72,68,75]
dept2 = [80,85,78,82,84]
dept3 = [60,62,58,65,63]

# Perform One-Way ANOVA
f_stat, p_value = stats.f_oneway(dept1, dept2, dept3)

print("F Statistic:", f_stat)
print("P Value:", p_value)

if p_value < 0.05:
    print("Reject Null Hypothesis: At least one group mean differs")
else:
    print("Fail to Reject Null Hypothesis: No significant difference")
