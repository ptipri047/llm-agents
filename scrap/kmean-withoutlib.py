import numpy as np
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation

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

    def initialize_data(self):
        xarray = np.empty((0, 2))
        for i in range(self.nbcluster):
            locx = i + 0.5*np.random.rand()
            locy = 5 * np.random.rand()
            loc=[locx, locy]
            scale = 0.2 + 0.1 * np.random.rand()
            x = np.random.normal(loc,scale,(self.nbofpoints,2))
            xarray = np.append(xarray, x, axis=0)
            
        self.X = xarray

    def euclidian_distance(self, point1, point2):
        return np.sqrt(np.sum((point1 - point2) ** 2))

    def init_centroids(self):
        centroids = np.empty((0, 2))
        for i in range(self.nbcluster):
            mini = np.min(self.X)
            maxi = np.max(self.X)
            x = mini + (maxi-mini)*np.random.random()
            y = mini + (maxi-mini)*np.random.random()
            
            #X = 10 * np.random.randn(1, 2) - 5
            centroids = np.append(centroids, [[x,y]], axis=0)
        self.centroids = centroids

    def kmean_run(self):
        # for loop for maxnbiters
        # assigner data points to centroids
        # recalculer mes centroids
        # sort de loop si mes centroids ne bougent plus trop
        
        self.data_point_index_history = []
        self.centroids_history = []

        for _ in range(self.maxnbiters):
            # assign data points to centroids
            x_centroids_index = self.assign_pointto_centroids()

            # recalculer mes centroids
            new_centroids = self.recompute_centroids(x_centroids_index)
            
            self.data_point_index_history.append(x_centroids_index)
            self.centroids_history.append(new_centroids)
            self.centroids = new_centroids


    def recompute_centroids(self, x_centroids_index):
        
        new_centroids = np.empty((0,2))
        
        for centroidnb, centroid in enumerate(self.centroids):
           centroid_new = np.mean([ point for pointindex,point in enumerate(self.X) 
                                if x_centroids_index[pointindex] == centroidnb],axis=0)
           
           if not isinstance(centroid_new, np.ndarray):
               centroid_new = centroid
           
           centroid_new = centroid_new.reshape((-1,2)) 
           
           new_centroids = np.append(new_centroids,centroid_new, axis=0)

        return new_centroids

    def visualize(self):
        # matplotlib.animation
        # visualiser mes clusters dans une animation
        
        # tracer mes datapoints [500,2]
        '''print(f'x shape: {self.X.shape}')
        x = self.centroids[:,0]
        y = self.centroids[:,1]
        plt.scatter(x,y, c='r')
        
        x = self.X[:,0]
        y = self.X[:,1]
        plt.scatter(x,y, c='b')
        
        plt.show()
        '''
        fig, ax = plt.subplots()
        x = self.centroids[:,0]
        y = self.centroids[:,1]
        centroidsc = ax.scatter(x,y, c=range(5), marker = 'X', s= 500)
        
        x = self.X[:,0]
        y = self.X[:,1]
        datasc = ax.scatter(x,y, c='b', alpha=0.6)
        
        astext = ax.text(0,0,'starting')
        
        def update(frame):
            centroids = self.centroids_history[frame]
            indexes = self.data_point_index_history[frame]
            
            centroidsc.set_offsets(centroids)
            datasc.set_color(plt.cm.tab10(indexes))
            centroidsc.set_color(plt.cm.tab10(range(5)))
            
            #astext.
            
            
            return centroidsc,datasc

        ani = FuncAnimation(fig, update, frames=self.maxnbiters, blit=True,interval=2000)
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
            
            # distances (5) [0.2,0.3,0.4,0.5,0.5]
            # distances.shape
            index = np.argmin(distances)
            x_centroids_index = np.append(x_centroids_index, [index])

        x_centroids_index = x_centroids_index.astype(np.integer)
        return x_centroids_index

    def __str__(self):
        return f"""nbclusters: {self.nbcluster}
                 data: {self.X[0:5,:]}
                 centroids: {self.centroids[:,:]}"""

    def test():
        print("test")
        pass


kmean = KMean(100, 5, 20, 0.001)
kmean.kmean_run()
kmean.visualize()
KMean.test()
print(kmean)
