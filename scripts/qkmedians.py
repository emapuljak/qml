import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import time
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool
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
        points: numpy.ndarray of shape (N, X)
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

def find_distance_matrix_quantum(XA, XB):
    XA = np.asarray(XA)
    XB = np.asarray(XB)

    sA = XA.shape
    sB = XB.shape
    if len(sA) != 2:
        raise ValueError('XA must be a 2-dimensional array.')
    if len(sB) != 2:
        raise ValueError('XB must be a 2-dimensional array.')
    if sA[1] != sB[1]:
        raise ValueError('XA and XB must have the same number of columns '
                         '(i.e. feature dimension.)')

    mA = sA[0]; mB = sB[0]; n = sA[1]
    dist_matrix = np.zeros((mA, mB))
    for i in range(mA):
        distance, _ = distc.DistCalc_DI(XA[i,:], XB[0])
        dist_matrix[i,:] = distance
    return dist_matrix
    
    
def geometric_median(X, eps=1e-5):
    if X.size==0: 
        print("For this class there is no points assigned!")
        return
    y = np.mean(X, 0)
    z=0
    while True:
        D = find_distance_matrix_quantum(X, [y])
        nonzeros = (D != 0)[:, 0]
        Dinv = 1 / D[nonzeros]
        Dinv_sum = np.sum(Dinv)
        W = Dinv / Dinv_sum
        T1 = np.sum(W * X[nonzeros], 0) #scaled sum of all points
        
        num_zeros = len(X) - np.sum(nonzeros) # number of points = y
        if num_zeros == 0: #then next median is scaled sum of all points
            y1 = T1
        elif num_zeros == len(X):
            return y
        else:
            R = (T1 - y) * Dinv_sum
            r = np.linalg.norm(R)
            gamma = 0 if r == 0 else num_zeros/r
            gamma = min(1, gamma)
            y1 = (1-gamma)*T1 + gamma*y
        
        # converge condition    
        dist_y_y1,_ = distc.DistCalc_DI(y, y1)
        if dist_y_y1 < eps:
            return y1
        y = y1 # next median is
        z+=1
        
def find_centroids_GM(points, cluster_labels, clusters=2):
    start_time = time.time()
    
    centroids = np.zeros([clusters, points.shape[1]])
    k = points.shape[1]
    for j in range(clusters):
        print(f'Searching centroids for cluster {j}')
        points_class_i = points[cluster_labels==j]
        median = geometric_median(points_class_i)
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
    cluster_label=[]
    distances=[]
    
    for i in range(n): # through all training samples
        dist=[]
        for j in range(k): # distance of each training example to each centroid
            temp_dist, _ = distc.DistCalc(points[i,:], centroids[j,:], shots_n=10000)
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
    cluster_label=[]
    distances=[]
    
    for i in range(n): # through all training samples
        dist=[]
        for j in range(k): # distance of each training example to each centroid
            temp_dist, _ = distc.DistCalc_AmplE(points[i,:], centroids[j,:], shots_n=10000)
            dist.append(temp_dist)
        cluster_index = m.duerr_hoyer_algo(dist)
        #cluster_index = np.argmin(dist)
        cluster_label.append(cluster_index)
        distances.append(dist)
    print("Find Cluster Labels ---> %s seconds ---" % (time.time() - start_time))
    return np.asarray(cluster_label), np.asarray(distances)

def find_nearest_neighbour_DI(points, centroids):
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
    cluster_label=[]
    distances=[]
    
    for i in range(n): # through all training samples
        dist=[]
        for j in range(k): # distance of each training example to each centroid
            temp_dist, _ = distc.DistCalc_DI(points[i,:], centroids[j,:], shots_n=10000)
            dist.append(temp_dist)
        cluster_index = m.duerr_hoyer_algo(dist)
        #cluster_index = np.argmin(dist)
        cluster_label.append(cluster_index)
        distances.append(dist)
    print("Find Cluster Labels ---> %s seconds ---" % (time.time() - start_time))
    return np.asarray(cluster_label), np.asarray(distances)
