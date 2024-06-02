# Importing libraries and the dataset for the project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('Market_Basket_Optimisation.csv', header=None)
transaction = []
for i in range(0, 7501):
    transaction.append([str(df.values[i, j])for j in range(0, 20)])
  #Applying the apriori
from apyori import apriori
rules = apriori(transactions = transaction, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)
print(rules)

results = list(rules)
print(results)
### Putting the results well organised into a Pandas DataFrame and display the results

