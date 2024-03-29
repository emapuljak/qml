""" Implementation of KMeans from @hounslow at GitHub """
import numpy as np
import scripts.util as u

class Kmeans:

    def __init__(self, k):
        self.k = k
        self.means = []

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        means = np.zeros((self.k, D))
        for kk in range(self.k):
            i = np.random.randint(N)
            means[kk] = X[i]

        while True:
            y_old = y

            # Compute euclidean distance to each mean
            dist2 = u.euclidean_dist_squared(X, means)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)

            # Update means
            for kk in range(self.k):
                means[kk] = X[y==kk].mean(axis=0)

            changes = np.sum(y != y_old)
            # print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                break

        self.means = means

    def predict(self, X):
        means = self.means
        dist2 = u.euclidean_dist_squared(X, means)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1)

    def error(self, X):
        N, D = X.shape
        means = self.means
        closest_mean_indexes = self.predict(X)

        error = 0
        for i in range(means.shape[0]):
            error += np.sum(utils.euclidean_dist_squared(X[closest_mean_indexes==i], means[[i]]))
        return error