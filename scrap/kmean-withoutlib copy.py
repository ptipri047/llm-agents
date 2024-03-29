import numpy as np
from matplotlib import pyplot as plt


class KMean:
    def __init__(
        self, nbofpoints: int, nbcluster: int, maxnbiters: int, stopvalue: float
    ):
        print("initialize")
        #np.random.seed(5)
        self.nbofpoints = nbofpoints
        self.nbcluster = nbcluster
        self.maxnbiters = maxnbiters
        self.stopvalue = stopvalue

        self.initialize_data()
        self.init_centroids()

    # pti
    def initialize_data(self):
        xarray = np.empty((0, 2))
        for i in range(self.nbcluster):
            X = np.random.normal(
                (i + np.random.rand(), i + np.random.rand()),
                (0.5 * np.random.rand(), 0.5 * np.random.rand()),
                (self.nbofpoints, 2),
            )
            xarray = np.append(xarray, X, axis=0)
        self.X = xarray
        #plt.scatter(self.X[:, 0], self.X[:, 1])
        # plt.show()

    def euclidian_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def init_centroids(self):
        #pti
        avgdata = np.mean(self.X, axis=0)
        self.centroids = np.random.normal(avgdata, 1, (self.nbcluster, 2))
        '''plt.scatter(
            self.centroids[:, 0],
            self.centroids[:, 1],
            c=range(5),
            s=[50] * 5,
            marker="X",
        )
        plt.show()'''

    def kmean_run(self):
        # for loop for maxnbiters
        # assigner data points to centroids
        # recalculer mes centroids
        # sort de loop si mes centroids ne bougent plus trop

        self.data_point_index_history = []
        self.centroids_history = []

        for niter in range(self.maxnbiters):
            # assign data points to centroids
            x_centroids_index = self.assign_pointto_centroids()

            # recalculer mes centroids
            new_centroids = self.recompute_centroids(x_centroids_index)

            self.data_point_index_history.append(x_centroids_index)

            # pti
            # self.centroids_history.append(new_centroids)
            self.centroids_history.append(self.centroids)
            oldcentroids = self.centroids.copy()
            self.centroids = new_centroids

            updatemax = np.max(np.abs(oldcentroids - new_centroids))

            if updatemax < self.stopvalue:
                print("reached precision")
                return

    # pti
    def recompute_centroids(self, x_centroids_index):
        new_centroids = np.empty((0, 2))

        for centroidnb, centroid in enumerate(self.centroids):
            centroid_new = np.mean(
                [
                    point
                    for pointindex, point in enumerate(self.X)
                    if x_centroids_index[pointindex] == centroidnb
                ],
                axis=0,
            )

            if isinstance(centroid_new, np.floating):
                centroid_new = centroid
            centroid_new = centroid_new.reshape((-1, 2))

            new_centroids = np.append(new_centroids, centroid_new, axis=0)

        return new_centroids

    def visualize(self):
        import matplotlib as mpl
        from matplotlib.animation import FuncAnimation

        fig, ax = plt.subplots()
        ax.set_title("my plot")
        thetext = ax.text(0,0,'starting')

        centroids_plot = ax.scatter(
            np.empty(5), np.empty(5), c=plt.cm.tab10(range(5)), s=[200] * 5, marker="X"
        )
        nbrows = self.X.shape[0]
        data_plot = ax.scatter(self.X[:, 0], self.X[:, 1], s=[5] * nbrows, alpha=0.5)
        ax.autoscale()

        def update(frame):
            print(f"{frame}")
            thetext.set_text(f'iteration:{frame}')
            centroids = self.centroids_history[frame]
            data_cluster_index = self.data_point_index_history[frame]
            data_cluster_index = data_cluster_index.astype(int)
            data_plot.set_color(plt.cm.tab10(data_cluster_index))
            centroids_plot.set_offsets(centroids)
            centroids_plot.set_color(plt.cm.tab10(range(5)))

            return thetext,centroids_plot, data_plot

        nbiters = len(self.data_point_index_history)
        ani = FuncAnimation(
            fig, update, frames=nbiters, interval=3000, repeat=True, blit=True
        )
        plt.show()

    def assign_pointto_centroids(self):
        # sortir un tableau avec pour chaque datapoint l'index du centroid dont il est le plus proche

        x_centroids_index = np.empty((0))
        for apoint in self.X:
            distances = np.empty((0))
            for acentroid in self.centroids:
                dist = self.euclidian_distance(apoint, acentroid)
                distances = np.append(distances, [dist], axis=0)
            # min
            index = np.argmin(distances)
            x_centroids_index = np.append(x_centroids_index, [index])

        return x_centroids_index

    def __str__(self):
        return f"""nbclusters: {self.nbcluster}
                 data: {self.X[0:5,:]}
                 centroids: {self.centroids[:,:]}"""

    def test():
        print("test")
        pass

from matplotlib import colormaps
#print(list(colormaps))
#print(plt.cm.tab10)

kmean = KMean(100, 5, 20, 0.001)
kmean.kmean_run()
kmean.visualize()
KMean.test()
print(kmean)
