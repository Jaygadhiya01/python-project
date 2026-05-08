# ==========================================
# ASSOCIATION RULE LEARNING - APRIORI
# ==========================================

# Step 1: Import libraries
import pandas as pd

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

# Step 2: Create transaction dataset
data = {
    "Bread":  [1,1,0,1,1],
    "Butter": [1,1,1,0,1],
    "Milk":   [1,0,1,1,1],
    "Eggs":   [0,1,1,1,1]
}

df = pd.DataFrame(data)

print("===== TRANSACTION DATA =====")
print(df)

# Step 3: Apply Apriori
frequent_items = apriori(df, min_support=0.4, use_colnames=True)

print("\n===== FREQUENT ITEMSETS =====")
print(frequent_items)

# Step 4: Generate association rules
rules = association_rules(frequent_items,
                          metric="confidence",
                          min_threshold=0.5)

print("\n===== ASSOCIATION RULES =====")
print(rules)