import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
#%matplotlib inline
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes

data = pd.read_csv("Mall_Customers.csv")
#print(data.head())
#print(data.describe())
X = data[["Annual Income (k$)", "Spending Score (1-100)"]]
plt.scatter(X["Annual Income (k$)"], X["Spending Score (1-100)"], c="black")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()
kmeans = KMeans(n_clusters=5, init='k-means++')
kmeans.fit(X)
print(kmeans.inertia_)

#Fititng multiple kmeans algorithms and storing the values in an empty list
SSE = []
for cluster in range(1,20):
    kmeans = KMeans(n_clusters=cluster, init='k-means++')
    kmeans.fit(X)
    SSE.append(kmeans.inertia_)

#converting results into a DataFrame and plotting them

frame = pd.DataFrame({'cluster':range(1,20), 'SSE':SSE})
plt.figure(figsize=(12,6))
plt.plot(frame['cluster'], frame['SSE'],marker = 'o')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

#KMeans using 6 clusters and k-means++ initialization

kmeans = KMeans(n_clusters=6, init='k-means++')
kmeans.fit(X)
pred = kmeans.predict(X)

#Value count of points
frame = pd.DataFrame(X)
frame['cluster'] = pred
print(frame['cluster'].value_counts())
color = ['red', 'blue', 'green', 'yellow', 'black', 'cyan']
K = 6

for k in range(K):
    data = X[X['cluster']==k]
    plt.scatter(data['Annual Income (k$)'], data['Spending Score (1-100)'], c=color[k])
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.show()


# KM = KModes(n_clusters=2, init='Huang', n_init=1, verbose=1)
# clusters = KM.fit_predict(data)
# print(KM.cluster_centroids_)
# print(clusters)
# data.insert(0,"Cluster",clusters,True)
# print(data)
