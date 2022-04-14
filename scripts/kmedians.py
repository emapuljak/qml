""" Implementation of KMedians from @hounslow at GitHub """

import numpy as np
import scripts.util as u
from scipy.spatial import distance

def euclidean_dist_squared(a, b):
    
    #np.sum(a**2,axis=1) + np.sum(b**2, axis=1) - 2 * np.dot(a,b.T)
    # sum1 = np.sum(a**2)
    # sum2 = np.sum(b**2, axis=1)
    #return np.sum(a**2) + np.sum(b**2, axis=1) - 2 * np.dot(a,b.T)
    return np.sum(a**2, axis=1)[:,None] + np.sum(b**2, axis=1)[None] - 2 * np.dot(a,b.T)

class Kmedians:

    def __init__(self, k):
        self.k = k
        self.medians = []
        self.loss = []
        self.tolerance=1e-3

    def fit(self, X):
        N, D = X.shape
        y = np.ones(N)

        # medians = np.zeros((self.k, D))
        # for kk in range(self.k):
        #     i = np.random.randint(N)
        #     medians[kk] = X[i]
        # indexes = np.random.randint(X.shape[0], size=self.k)
        # medians = X[indexes]
        #np.save(f'/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/centroids/centroids_random_start_24032022_1_Durr_DI_AE_500.npy', medians)
        medians=np.load('/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/centroids/centroids_random_start_24032022_1_Durr_DI_AE_500.npy')
        iteration=0
        
        #new_medians = np.zeros((self.k, D))
        while True:
            y_old = y

            # Compute euclidean distance to each mean
            #dist2 = euclidean_dist_squared(X, medians)
            
            ### CHECK DIFF EUCLIDIAN DISTANCE IMPLEMENTATIONSS
            dist2=[]
            for i in range(N): # through all training samples
                d=[]
                for j in range(self.k): # distance of each training example to each centroid
                    temp_dist = u.euclidean_dist_squared(X[i,:], medians[j,:]) # returning back one number for all latent dimensions!
                    d.append(temp_dist)
                dist2.append(d)
            dist2 = np.array(dist2)
            dist2[np.isnan(dist2)] = np.inf
            y = np.argmin(dist2, axis=1)
            # Update medians
            for kk in range(self.k):
                # calculate mean of all samples assigned to cluster
                # to calculate the new cluster center
                if X[y == kk].shape[0] > 0:
                    medians[kk] = np.median(X[y==kk], axis=0)

            changes = np.sum(y != y_old)
            self.loss.append(np.linalg.norm(y - y_old))
            
            # if np.linalg.norm(medians - new_medians) < self.tolerance:
            #     medians=new_medians
            #     print(f"KMedians converged after {iteration+1} iterations.")
            #     break
            #print('Running K-means, changes in cluster assignment = {}'.format(changes))

            # Stop if no point changed cluster
            if changes == 0:
                print(f"KMedians converged after {iteration+1} iterations.")
                break
            #medians=new_medians
            iteration+=1

        self.medians = medians

    def predict(self, X):
        medians = self.medians
        dist2 = euclidean_dist_squared(X, medians)
        dist2[np.isnan(dist2)] = np.inf
        return np.argmin(dist2, axis=1), dist2

    def error(self, X):
        N, D = X.shape
        medians = self.medians
        closest_median_indexes = self.predict(X)

        error = 0
        for i in range(medians.shape[0]):
            error += np.sum(euclidean_dist_squared(X[closest_median_indexes==i], medians[[i]]))
        return error