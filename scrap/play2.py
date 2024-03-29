"""
write me the code, implementing a k-mean clustering algorithm.
The code should also generate the training dataset. I have no csv file.

i do not want to use any library for k-mean. Code should be pure python for k-mean.
i want to visualize each iteration of the k-mean training.
For visualization use  python class matplotlib.animation to create an animated gif, and save the result in a file in a temporary director called temp under ./ folder.
If directory ./temp does not exist create it. empty ./temp directory before using it
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import os
from os import path
from functools import partial


def generate_data(n_samples=1000, n_clusters=3):
    #np.random.seed(random_state)
    x = np.concatenate(
        [
            np.random.normal(-5, 1, n_samples // 3),
            np.random.normal(5, 1, n_samples // 3),
            np.random.normal(0, 0.5, n_samples // 3),
        ]
    )
    y = np.concatenate(
        [
            np.random.normal(-5, 1, n_samples // 3),
            np.random.normal(5, 1, n_samples // 3),
            np.random.normal(0, 0.5, n_samples // 3),
        ]
    )
    return np.array(list(zip(x, y))).reshape(-1, 2)


# pti
def initialize_centroids(data, k):
    return 10 * np.random.random_sample((k, 2)) - 2
    #return 10 * np.random.random_sample((k, 2)) - 5
    # return data[np.random.choice(data.shape[0], k, replace=False)]


def assign_clusters(data1, centroids):
    distances = np.array([np.linalg.norm(data1 - c, axis=1) for c in centroids])
    return np.argmin(distances, axis=0)


def update_centroids(data1, clusters, k):
    return np.array([data1[clusters == i].mean(axis=0) for i in range(k)])


# pti
centroidlist = []
clusterlist = []


def k_means(data, k, max_iter=100, tol=1e-4):
    centroids = initialize_centroids(data, k)
    print(centroids.shape)
    for i in range(max_iter):
        old_centroids = centroids.copy()
        clusters = assign_clusters(data, centroids)

        # pti
        centroidlist.append(centroids)
        clusterlist.append(clusters)

        centroids = update_centroids(data, clusters, k)
        if np.linalg.norm(centroids - old_centroids) < tol:
            break
    return clusters, centroids


# pti
def animate_k_means(data, k, save_gif=False):
    fig, ax = plt.subplots()
    currentcentroid = centroidlist[0]
    scat = ax.scatter(data[:, 0], data[:, 1], c="b", alpha=0.5)
    scat_centroids = ax.scatter(
        currentcentroid[:, 0], currentcentroid[:, 1], c="r", s=100
    )
    stt = ax.text(2.5, -10, 'starting...', fontsize='large')    


    # def update(i):
    def update(i):
        currentcentroid = centroidlist[i]
        currentcluster = clusterlist[i]
        # print(f"iter {i} - {currentcentroid}")
        scat.set_color(plt.cm.tab10(currentcluster))
        scat_centroids.set_offsets(currentcentroid)
        st = f"iteration:{i}"
        stt.set_text(st)
        

        return scat, scat_centroids

    anim = animation.FuncAnimation(
        ##fig, update, frames=max_iter, interval=100, blit=True
        fig,
        update,
        # partial(update, centroids=centroids, data=data),
        frames=len(centroidlist),
        interval=2000,
    )

    print("came here")
    if save_gif:
        if not path.exists("temp"):
            os.makedirs("temp")
        anim.save(path.join("temp", "k_means.gif"), fps=1)
    # plt.draw()
    plt.show()
    print("over")


data = generate_data()
k = 3
clusters, centroids = k_means(data, k)
animate_k_means(data, k, save_gif=True)
