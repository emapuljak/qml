import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import time
from multiprocessing.pool import ThreadPool
from time import time as ts
from qibo.models import Circuit
from qibo import gates
from qiskit.visualization import plot_histogram
from qiskit import QuantumCircuit, execute, Aer
from qiskit.tools.jupyter import *
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit, ClassicalRegister, QuantumRegister
from qiskit_textbook.tools import vector2latex

import scripts.qkmeans as qkm
import scripts.minimization as m
import scripts.distance_calc as distc
import sys
sys.path.append('../')
sys.path.append('../../')
import utils as u
    
global points_class_i

def initialize_centroids(points, k):
    """
    Randomly initialize centroids of data points.
    Args:
        points: array of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space - number of features
        k: int - number of clusters
    """
    indexes = np.random.randint(points.shape[0], size=k)
    return points[indexes]

def sum_distances(points, center):
    S_i = 0
    for point in points[:]:
        dist, _ = distc.DistCalc(point, center, points.shape[1])
        S_i += dist
    return S_i

def find_distance_matrix_quantum(points, centroid, device_name):
    """ 
    Modified version of scipy.spatial.distance.cdist() function.
    Args:
        points: array of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space - number of features
        centroid: array of shape (1, X)
    """
    
    points = np.asarray(points)
    centroid = np.asarray(centroid)

    n_features = points.shape[1]
    n_events = points.shape[0]
    
    if points.shape[1] != centroid.shape[1]:
        raise ValueError('Points and centroid need to have same number of features.')

    dist_matrix = np.zeros((n_events, centroid.shape[0]))
    for i in range(n_events):
        distance, _ = distc.DistCalc_DI(points[i,:], centroid[0], device_name)
        dist_matrix[i,:] = distance
    return dist_matrix
    
    
def geometric_median(points, median, eps=1e-6, device_name='/GPU:0'):
    """
    Implementation from Reference - DOI: 10.1007/s00180-011-0262-4
    Args:
        points: array of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space - number of features
    """
    
    if points.size==0: 
        print("For this class there is no points assigned!")
        return
    
    #median = np.mean(points, 0) # starting median

    while True:
        D = find_distance_matrix_quantum(points, [median], device_name)
        nonzeros = (D != 0)[:, 0]
        Dinv = 1 / D[nonzeros]
        Dinv_sum = np.sum(Dinv)
        W = Dinv / Dinv_sum
        T1 = np.sum(W * points[nonzeros], 0) #scaled sum of all points - Eq. (7) in Ref.
        
        num_zeros = len(points) - np.sum(nonzeros) # number of points = y
        if num_zeros == 0: #then next median is scaled sum of all points
            new_median = T1
        elif num_zeros == len(points):
            return median
        else:
            R = (T1 - median) * Dinv_sum # Eq. (9)
            r = np.linalg.norm(R)
            gamma = 0 if r == 0 else num_zeros/r
            gamma = min(1, gamma) # Eq. (10)
            new_median = (1-gamma)*T1 + gamma*median # Eq. (11)
        
        # converge condition    
        dist_med_newmed,_ = distc.DistCalc_DI(median, new_median, device_name=device_name)
        if dist_med_newmed < eps:
            return new_median
        median = new_median # next median
        
def find_centroids_GM(points, cluster_labels, start_centroids, clusters=2):
    """
    Args:
        points: array of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space - number of features
        cluster_labels: array of shape (N,) - cluster labels assigned to each data point
        clusters: int - number of clusters
    """
    start_time = time.time()
    
    centroids = np.zeros([clusters, points.shape[1]])
    k = points.shape[1]
    for j in range(clusters):
        print(f'Searching centroids for cluster {j}')
        points_class_i = points[cluster_labels==j]
        median = geometric_median(points_class_i, start_centroids[j, :])
        centroids[j,:] = median
        print(f'Found for cluster {j}')
    print("MedianCalc ---> %s seconds ---" % (time.time() - start_time))
    return np.array(centroids)

def SA_find_centroids(points, cluster_labels, clusters=2):
    start_time = time.time()
    
    centroids = np.zeros([clusters, points.shape[1]])
    k = points.shape[1]
    epsilon = 1e-1
    for j in range(clusters):
        #print(f'Searching centroids for cluster {j}')
        points_class_i = points[cluster_labels==j]
        median = np.mean(points, axis=0)
        min_dist = sum_distances(points_class_i, median)
        step = np.std(points_class_i)
        directions = points_class_i/np.linalg.norm(points_class_i)
        while step > epsilon:
            improved = False
            for i in range(directions.shape[0]):
                temp_median = median + directions[i,:]*step
                temp_min_dist = sum_distances(points_class_i, temp_median)
                if temp_min_dist < min_dist:
                    min_dist = temp_min_dist
                    #print(f'min distance: {min_dist}')
                    median = temp_median
                    improved = True
                    break
                    
            if not improved:
                step = step/2
        centroids[j,:] = median
        print(f'Found for cluster {j}')
    print("MedianCalc ---> %s seconds ---" % (time.time() - start_time))
    return np.array(centroids)

def proc(point):
    S_i = 0
    for p in points_class_i[:]:
        if np.array_equal(p, point): continue
        dist, _ = distc.DistCalc(point, p, points_class_i.shape[1])
        S_i += dist
    return S_i

def find_centroids(points, cluster_labels, clusters=2):
    #print("Find new centroids")
    """
    Find new cluster centroids by calculating the mean of data points assigned to specific cluster.
    Args:
        points: numpy.ndarray of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space - number of features
        cluster_labels: numpy.ndarray of shape (N,)
    Returns:
        new cluster centroids for points
    """
    import time
    from tqdm import tqdm
    start_time = time.time()
    
    centroids = np.zeros([clusters, points.shape[1]])
    k = points.shape[1]
    for i in range(clusters):
        print(f'Searching centroids for cluster {i}')
        global points_class_i
        points_class_i = points[cluster_labels==i]
        N = points_class_i.shape[0]
        sums=[]
        with ThreadPool(processes = 36) as pool:
            sums = list(tqdm(pool.imap_unordered(proc, points_class_i[:]), total=N))
        sums = np.array(sums)
        #min_sum_index = np.argmin(sums)
        min_sum_index = m.duerr_hoyer_algo(sums)
        centroids[i,:] = points_class_i[min_sum_index, :]
        print(f'Found for cluster {i}')
    print("MedianCalc ---> %s seconds ---" % (time.time() - start_time))
    return np.array(centroids)

def find_nearest_neighbour(points, centroids):
    import time
    start_time = time.time()
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
        #print(f'Point for distance: {points[i,:]}')
        for j in range(k): # distance of each training example to each centroid
            #print(f'Centroid for distance: {centroids[j,:]}')
            temp_dist, _ = distc.DistCalc(points[i,:], centroids[j,:], shots_n=10000) # returning back one number for all latent dimensions!
            dist.append(temp_dist)
        cluster_index = m.duerr_hoyer_algo(dist)
        #cluster_index = np.argmin(dist)
        cluster_label.append(cluster_index)
        distances.append(dist)
    print("Find Cluster Labels ---> %s seconds ---" % (time.time() - start_time))
    return np.asarray(cluster_label), np.asarray(distances)

def find_nearest_neighbour_AmplE(points, centroids):
    import time
    start_time = time.time()
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
        #print(f'Point for distance: {points[i,:]}')
        for j in range(k): # distance of each training example to each centroid
            #print(f'Centroid for distance: {centroids[j,:]}')
            temp_dist, _ = distc.DistCalc_AmplE(points[i,:], centroids[j,:], shots_n=10000) # returning back one number for all latent dimensions!
            dist.append(temp_dist)
        #cluster_index = m.duerr_hoyer_algo(dist)
        cluster_index = np.argmin(dist)
        cluster_label.append(cluster_index)
        distances.append(dist)
    print("Find Cluster Labels ---> %s seconds ---" % (time.time() - start_time))
    return np.asarray(cluster_label), np.asarray(distances)

def find_nearest_neighbour_DI(points, centroids, device_name='/GPU:0'):
    import time
    start_time = time.time()
    """
    Args:
        points: array of shape (N, X)
                    N = number of samples,
                    X = dimension of latent space;
        centroids: array of shape (N, X)
    Returns:
        cluster labels: array of shape (N,) specifying to which cluster each point is assigned
        distances: array of shape (N,) specifying distances to nearest cluster for each point
    """
    
    n = points.shape[0]
    num_features = points.shape[1]
    k = centroids.shape[0] # number of centroids
    cluster_label=[]
    distances=[]
    
    for i in range(n): # through all training samples
        dist=[]
        for j in range(k): # distance of each training example to each centroid
            temp_dist, _ = distc.DistCalc_DI(points[i,:], centroids[j,:], device_name, shots_n=10000) # returning back one number for all latent dimensions!
            dist.append(temp_dist)
        #cluster_index = m.duerr_hoyer_algo(dist)
        cluster_index = np.argmin(dist)
        cluster_label.append(cluster_index)
        distances.append(dist)
    print("Find Cluster Labels ---> %s seconds ---" % (time.time() - start_time))
    return np.asarray(cluster_label), np.asarray(distances)


def find_nearest_neighbour_NegRot(points, centroids):
    import time
    start_time = time.time()
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
        qc, counts = m.neg_rotations(points[i, :], centroids)
        print(counts)
        #cluster_index = np.argmin(dist)
        #cluster_label.append(cluster_index)
        #distances.append(dist)
    print("Find Cluster Labels ---> %s seconds ---" % (time.time() - start_time))
    return np.asarray(cluster_label), np.asarray(distances)