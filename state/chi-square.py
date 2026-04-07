import pandas as pd
from scipy.stats import chi2_contingency

# Contingency table
# Rows: Gender (Male/Female)
# Columns: Course Preference (Online/Offline)
data = [[30, 20],
        [25, 25]]

chi2, p, dof, expected = chi2_contingency(data)

print("Chi-Square Statistic:", chi2)
print("P Value:", p)
print("Expected Frequencies:\n", expected)

if p < 0.05:
    print("Reject Null Hypothesis: Variables are related")
else:
    print("Fail to Reject Null Hypothesis: Variables are independent")
