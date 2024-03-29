import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate random dataset
np.random.seed(10)
datatapoint = np.random.rand(100, 2) * 10 -5

# Define number of clusters
num_clusters = 5

# Apply K-means clustering algorithm
kmeans = KMeans(n_clusters=num_clusters)
kmeans.fit(datatapoint)
centroids = kmeans.cluster_centers_
labels = kmeans.labels_

# Visualize the clusters
colors = ['b', 'g', 'r', 'c', 'm']
plt.figure(figsize=(8, 6))

for i in range(num_clusters):
    points = np.array([datatapoint[j] for j in range(len(datatapoint)) if labels[j] == i])
    plt.scatter(points[:, 0], points[:, 1], s=7, c=colors[i], label=f'Cluster {i+1}')

plt.scatter(centroids[:, 0], centroids[:, 1], marker='o', s=200, c='k', label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('K-means Clustering')
plt.legend()
plt.show()

