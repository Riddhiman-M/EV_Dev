import csv
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin
from sklearn.metrics import davies_bouldin_score
from sklearn import metrics
from scipy.spatial.distance import cdist

# Path where the excel file containing the data is stored
store = pd.read_excel(r'C:/Users/Riddhiman Moulick/IIT_Kharagpur/pythonProject/Electric_Vehicle/Data.xlsx')

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

distortions = []
inertia = []
db_index = []

for k in range(2, 14):
   kmeans = KMeans(k)
   kmeans.fit(coord, sample_weight=wt)
   cluster_data = kmeans.fit_predict(coord, sample_weight=wt)

   index = davies_bouldin_score(coord, cluster_data)
   print(index, sep='\n')
   db_index.append(index)
   distortions.append(sum(np.min(cdist(coord, kmeans.cluster_centers_,
                                       'euclidean'), axis=1)) / coord.shape[0])
   inertia.append(kmeans.inertia_)


plt.plot(range(2, 14), db_index, 'bx-')
plt.xlabel('Values of K')
plt.ylabel('DB index')
plt.title('The Elbow Method using Distortion')
plt.show()

# Taking User input for number of clusters to be formed
num = int(input())
kmeans = KMeans(num)
# In the above lines num is the number of Charging Stations we want to set up

# sample_weight has been initialized to use 'Weighted' K-Means Clustering
kmeans.fit(coord, sample_weight=wt)
clusters_weighted = kmeans.fit_predict(coord, sample_weight=wt)

# db_index = davies_bouldin_score(coord, clusters_weighted)
# print(db_index)

plt.scatter(x, y, c=clusters_weighted, cmap='rainbow')

plt.title('Optimum Charging Station Locations')
plt.xlabel('X-coord')
plt.ylabel('Y-coord')
centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=100, alpha=0.5)
plt.text(3.6, 28, "Number of data points = " + str(len(x)), fontsize=10,
         bbox=dict(boxstyle="Square", alpha=0.05))

with open("Points_Obtained.csv", 'w') as file:
    cwriter = csv.writer(file)
    fields = ['X', 'Y']
    cwriter.writerow(fields)
    for i in range(len(centers[:, 0])):
        arr = [round(centers[i, 0], 2), round(centers[i, 1], 2)]
        cwriter.writerow(arr)

file.close()

# Marking co-ordinates of centers on graph
for i, j in centers:
   plt.text(i, j+0.5, '({}, {})'.format(round(i, 2), round(j, 2)))

plt.show()
