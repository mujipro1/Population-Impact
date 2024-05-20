import pandas as pd
import numpy as np
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

data = pd.read_excel("Refined/CleanData.xlsx")

# Convert data to DataFrame
df = pd.DataFrame(data)

# Define bins and labels for discretization
bins_total = [0, 50000, 100000, np.inf]
labels_total = ['Low', 'Medium', 'High']

bins_male = [0, 25000, 50000, np.inf]
labels_male = ['Low', 'Medium', 'High']

bins_female = [0, 25000, 50000, np.inf]
labels_female = ['Low', 'Medium', 'High']

bins_ratio = [0, 115, 120, np.inf]
labels_ratio = ['Low', 'Medium', 'High']

bins_age = [0, 18, 20, np.inf]
labels_age = ['Low', 'Medium', 'High']

bins_increase = [0, 2.5, 3, np.inf]
labels_increase = ['Low', 'Medium', 'High']

bins_life = [0, 50, 60, np.inf]
labels_life = ['Low', 'Medium', 'High']

bins_inflation = [0, 2, 5, np.inf]
labels_inflation = ['Low', 'Medium', 'High']

bins_gdp = [0, 5e9, 1e10, np.inf]
labels_gdp = ['Low', 'Medium', 'High']

# Discretize the attributes
df['Total_Bin'] = pd.cut(df['Total'], bins=bins_total, labels=labels_total)
df['Male_Bin'] = pd.cut(df['Male'], bins=bins_male, labels=labels_male)
df['Female_Bin'] = pd.cut(df['Female'], bins=bins_female, labels=labels_female)
df['Ratio_Bin'] = pd.cut(df['Ratio'], bins=bins_ratio, labels=labels_ratio)
df['MedianAge_Bin'] = pd.cut(df['Median Age'], bins=bins_age, labels=labels_age)
df['IncreaseRate_Bin'] = pd.cut(df['Increase Rate'], bins=bins_increase, labels=labels_increase)
df['LifeExpectancy_Bin'] = pd.cut(df['Life Expectancy'], bins=bins_life, labels=labels_life)
df['Inflation_Bin'] = pd.cut(df['Inflation'], bins=bins_inflation, labels=labels_inflation)
df['GDP_Bin'] = pd.cut(df['GDP'], bins=bins_gdp, labels=labels_gdp)

# Create transactions
transactions = df[['Total_Bin', 'Male_Bin', 'Female_Bin', 'Ratio_Bin', 'MedianAge_Bin',
                   'IncreaseRate_Bin', 'LifeExpectancy_Bin', 'Inflation_Bin', 'GDP_Bin']]

transactions_list = transactions.apply(lambda x: list(x.dropna()), axis=1).tolist()

# Convert transactions to the format required by mlxtend
te = TransactionEncoder()
te_ary = te.fit(transactions_list).transform(transactions_list)
df_trans = pd.DataFrame(te_ary, columns=te.columns_)

# Apply apriori algorithm
frequent_itemsets = apriori(df_trans, min_support=0.1, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)

# Display the rules
print(rules)
