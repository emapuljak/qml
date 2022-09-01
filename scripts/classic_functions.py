import scripts.util as u
import numpy as np

def find_nearest_neighbour_classic(points, centroids):
    
    """
    Args:
        points: numpy.ndarray of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space;
        centroids: numpy.ndarray of shape (N, X)
    Returns:
        cluster_assignments: numpy.ndarray of shape (N, X) specifying to which cluster each feature is assigned
        distances: numpy.ndarray of shape (N, X) specifying distances to nearest cluster
    """
    
    n = points.shape[0]
    num_features = points.shape[1]
    k = centroids.shape[0] # number of centroids
    #cluster_label = np.zeros(n) # assignment to new centroids
    cluster_label=[]
    distances=[]
    
    for i in range(n): # through all training samples
        dist=[]
        for j in range(k): # distance of each training example to each centroid
            temp_dist = u.euclidean_dist(points[i,:], centroids[j,:]) # returning back one number for all latent dimensions!
            dist.append(temp_dist)
        cluster_index = np.argmin(dist)
        cluster_label.append(cluster_index)
        distances.append(dist)
    return np.asarray(cluster_label), np.asarray(distances)