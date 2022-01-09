import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin

# Path where the excel file containing the data is stored
store = pd.read_excel(r'C:\Users\Riddhiman Moulick\IIT Kharagpur\pythonProject\Electric_Vehicle\Data.xlsx')

x = store["X"].to_numpy(dtype='float')
y = store["Y"].to_numpy(dtype='float')
n = len(x)

wt = store["Wt"].to_numpy(dtype='float')

# Creating a scatter plot for the input data
plt.scatter(x, y)
plt.title("Data-points")
plt.xlabel("X-coordinates")
plt.ylabel("Y-coordinates")
plt.show()

coord = store.iloc[:, 1:3]

# Taking User input for number of clusters to be formed
num = int(input())
kmeans = KMeans(num)
# In the above lines num is the number of Charging Stations we want to set up

# sample_weight has been initialized to use 'Weighted' K-Means Clustering
kmeans.fit(coord, sample_weight=wt)
clusters_weighted = kmeans.fit_predict(coord, sample_weight=wt)

plt.scatter(x, y, c=clusters_weighted, cmap='rainbow')

plt.title('Optimum Charging Station Locations')
plt.xlabel('X-coord')
plt.ylabel('Y-coord')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)

# Marking co-ordinates of centers on graph
for i, j in centers:
   plt.text(i, j+0.5, '({}, {})'.format(round(i, 2), round(j, 2)))

plt.show()
