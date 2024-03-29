import time

import os

time.sleep(20)

os.abort()
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Generate random dataset
np.random.seed(0)
X = np.random.rand(100, 2)

# Apply k-means clustering
kmeans = KMeans(n_clusters=5, random_state=0).fit(X)

# Get the cluster labels
labels = kmeans.labels_

# Plot the results
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.show()
