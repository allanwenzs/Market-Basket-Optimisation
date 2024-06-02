#Importing the required libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Reading and processing dataset from csv file
df = pd.read_csv('market_Basket_OptimisationEclat.csv', header=None)
transactions=[]
for i in range(0, 7501):
	#pass the values
	transactions.append([str(df.values[i,j])for j in range(0,20)])
	transactions[:2]
## Visualising the results
print(transactions)

# applying Apriori model on the dataset
from apyori import apriori
rules = apriori(transactions=transactions, min_support=0.003, min_confidence=0.2, min_lift=3, min_length=2, max_features='2')
results = list(rules)
print(results)
### Putting the results well organised into a Pandas DataFrame and display the results
def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    confidences = [result[2][0][2] for result in results]
    lifts       = [result[2][0][3] for result in results]
    return list(zip(lhs, rhs, supports, confidences, lifts))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Left Hand Side', 'Right Hand Side', 'Support', 'Confidence', 'Lift'])
resultsinDataFrame
print('Data in organised format')
### Displaying the results sorted by descending lifts
print(resultsinDataFrame.nlargest(n = 10, columns = 'Lift'))
