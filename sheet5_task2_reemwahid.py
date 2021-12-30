# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 17:56:11 2021

@author: DELL
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#question 1 1.	Factorize the YearsExp feature and convert it to numbers in new col.
dataset = pd.read_csv("Wuzzuf_Jobs.csv")
dataset.head()
x = dataset.iloc[:,5 ].values
x
dataset["factorizedYearsExp"] = pd.factorize(dataset["YearsExp"])[0]
dataset.head()
# question 2   2.	Apply K-means for job title and companies.
dataset["Title"]= pd.factorize(dataset["Title"])[0]
dataset["Company"]= pd.factorize(dataset["Company"])[0]
dataset.head()
y = dataset.iloc[:, [0,1]].values
y
from sklearn.cluster import KMeans 
wcss = []
for i in range(1, 21):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(y)
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 21), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()
from kneed import KneeLocator
K = KneeLocator(range(1, 21), wcss, curve="convex", direction="decreasing")
print(f"the best number of clusters is {K.elbow}")
kmeans = KMeans(n_clusters= 5, init= 'k-means++', random_state=42)
y_kmeans = kmeans.fit_predict(y)
plt.subplots(figsize=(20, 6))
colors = ["red", "purple", "yellow", "teal","olive", "gray"]
for i in range(5):
    plt.scatter(y[y_kmeans == i, 0], y[y_kmeans == i, 1], s=100, c = colors[i], label = f'Cluster {i+1}')
#     plt.scatter(X[y_kmeans == i, 0], X[y_kmeans == i, 1], s=100, c=color_list[i], label=f'Cluster{i+1}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c=colors[5], label='Centroids')
plt.title('Clusters of Jobs')
plt.xlabel('Jobs')