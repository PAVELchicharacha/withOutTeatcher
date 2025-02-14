import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN

data = pd.read_csv('cluster_mpg.csv')

X = data[['cylinders', 'horsepower', 'weight']]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=3, random_state=42) 
kmeans_labels = kmeans.fit_predict(X_scaled)

agglo = AgglomerativeClustering(n_clusters=3)  
agglo_labels = agglo.fit_predict(X_scaled)

dbscan = DBSCAN(eps=0.5, min_samples=5) 
dbscan_labels = dbscan.fit_predict(X_scaled)

fig, axes = plt.subplots(1, 3, figsize=(18, 6))

axes[0].scatter(X.iloc[:, 0], X.iloc[:, 1], c=kmeans_labels, cmap='viridis', marker='o', edgecolor='k')
axes[0].set_title('K-Means Clustering')
axes[0].set_xlabel('cylinders')
axes[0].set_ylabel('horsepower')

axes[1].scatter(X.iloc[:, 0], X.iloc[:, 1], c=agglo_labels, cmap='plasma', marker='o', edgecolor='k')
axes[1].set_title('Agglomerative Clustering')
axes[1].set_xlabel('cylinders')
axes[1].set_ylabel('horsepower')

axes[2].scatter(X.iloc[:, 0], X.iloc[:, 1], c=dbscan_labels, cmap='inferno', marker='o', edgecolor='k')
axes[2].set_title('DBSCAN Clustering')
axes[2].set_xlabel('cylinders')
axes[2].set_ylabel('horsepower')

plt.tight_layout()
plt.show()