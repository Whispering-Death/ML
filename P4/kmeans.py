import numpy as np


class KMeans():

    '''
        Class KMeans:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float)
    '''

    def __init__(self, n_cluster, max_iter=100, e=0.0001):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x):
        '''
            Finds n_cluster in the data x
            params:
                x - N X D numpy array
            returns:
                A tuple
                (centroids a n_cluster X D numpy array, y a size (N,) numpy array where cell i is the ith sample's assigned cluster, number_of_updates an Int)
            Note: Number of iterations is the number of time you update the assignment
        ''' 
        assert len(x.shape) == 2, "fit function takes 2-D numpy arrays as input"
        np.random.seed(42)
        N, D = x.shape

        # TODO:
        # - comment/remove the exception.
        # - Initialize means by picking self.n_cluster from N data points
        # - Update means and membership until convergence or until you have made self.max_iter updates.
        # - return (means, membership, number_of_updates)

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        def compute_distortion(mu , x, r):
            N = x.shape[0]

            J= np.sum([np.sum((x[r==i]-mu[i])) for i in range(self.n_cluster)])

            return J/N


        mu = x[np.random.choice(N,self.n_cluster),:]

        r= np.zeros(N)

        J = compute_distortion(mu,x,r)

        n_iter =0

        while n_iter< self.max_iter:

            l2_norm  = np.sum(((x - np.expand_dims(mu,axis=1))**2), axis=2)
            r= np.argmin(l2_norm, axis=0)

            J_n = compute_distortion(mu, x, r)

            if np.absolute(J-J_n)<= self.e:
                break

            J= J_n

            mu_n = np.array([ np.mean(x[r==cluster_ind], axis=0) for cluster_ind in range(self.n_cluster) ])

            index = np.where(np.isnan(mu_n))
            mu_n[index] = mu[index]
            
            mu = mu_n
            
            n_iter += 1

        return (mu,r, n_iter)
        # DONOT CHANGE CODE BELOW THIS LINE

class KMeansClassifier():

    '''
        Class KMeansClassifier:
        Attr:
            n_cluster - Number of cluster for kmeans clustering (Int)
            max_iter - maximum updates for kmeans clustering (Int) 
            e - error tolerance (Float) 
    '''

    def __init__(self, n_cluster, max_iter=100, e=1e-6):
        self.n_cluster = n_cluster
        self.max_iter = max_iter
        self.e = e

    def fit(self, x, y):
        '''
            Train the classifier
            params:
                x - N X D size  numpy array
                y - (N,) size numpy array of labels
            returns:
                None
            Stores following attributes:
                self.centroids : centroids obtained by kmeans clustering (n_cluster X D numpy array)
                self.centroid_labels : labels of each centroid obtained by 
                    majority voting ((N,) numpy array) 
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"
        assert len(y.shape) == 1, "y should be a 1-D numpy array"
        assert y.shape[0] == x.shape[0], "y and x should have same rows"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the classifier
        # - assign means to centroids
        # - assign labels to centroid_labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        kmeans = KMeans(self.n_cluster, self.max_iter, self.e)

        centroids, membership, n_iter = kmeans.fit(x)

        voting = []

        for k in range(self.n_cluster):
            voting.append({})

        for y_i, r_i in zip(y,membership):

            voting[r_i][y_i]= voting[r_i].get(y_i,0)+1

        centroid_labels=[]

        for vote in voting:

            if not vote:
                centroid_labels.append(0)
            else:
                centroid_labels.append(max(vote, key=vote.get))

        centroid_labels= np.array(centroid_labels)


        # DONOT CHANGE CODE BELOW THIS LINE

        self.centroid_labels = centroid_labels
        self.centroids = centroids

        assert self.centroid_labels.shape == (self.n_cluster,), 'centroid_labels should be a numpy array of shape ({},)'.format(
            self.n_cluster)

        assert self.centroids.shape == (self.n_cluster, D), 'centroid should be a numpy array of shape {} X {}'.format(
            self.n_cluster, D)

    def predict(self, x):
        '''
            Predict function

            params:
                x - N X D size  numpy array
            returns:
                predicted labels - numpy array of size (N,)
        '''

        assert len(x.shape) == 2, "x should be a 2-D numpy array"

        np.random.seed(42)
        N, D = x.shape
        # TODO:
        # - comment/remove the exception.
        # - Implement the prediction algorithm
        # - return labels

        # DONOT CHANGE CODE ABOVE THIS LINE
        
        l2_norm  = np.sum(((x - np.expand_dims(self.centroids,axis=1))**2), axis=2)
        r= np.argmin(l2_norm, axis=0)

        labels = self.centroid_labels[r]
        # DONOT CHANGE CODE BELOW THIS LINE
        return labels

