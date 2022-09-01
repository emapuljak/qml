import sys
sys.path.append('../')
sys.path.append('../../')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import math
import h5py
from datetime import datetime
import qibo
qibo.set_backend("tensorflow")
import scripts.qkmeans as qkm
import scripts.minimization as m
import scripts.qkmedians as qkmed
import scripts.oracle as o
import utils as u
import plots as p

qibo.set_device("/GPU:0")
    
# read QCD predicted data (test - SIDE)
# set dataset and sample to use
sample_id = 'dijet'
qcd_id = 'qcd_sig'
stacked_jets_signal, _ = dare.load_inference_dataset_qcd(sample_id, qcd_id)

num_samples = 500
data_train = stacked_jets_signal[:num_samples]

# TRAIN Q-MEDIANS
k = 2 # number of clusters
centroids = qkmed.initialize_centroids(data_train, k)   # Intialize centroids
tolerance=1e-3

# run k-medians algorithm
i = 0
while True:
    cluster_label, _ = qkmed.find_nearest_neighbour_DI(data_train,centroids)       # find nearest centers
    print(f'Found cluster assignments for iteration: {i+1}')
    new_centroids = qkmed.find_centroids_GM(data_train,cluster_label, clusters=k)               # find centroids

    if np.linalg.norm(centroids - new_centroids) < tolerance:
        centroids = new_centroids
        print(f"Converged after {i+1} iterations.")
        break
    print(f'Iteration: {i+1}')
    i += 1     
    centroids = new_centroids
    if i == 150: 
        print('QKmedians stopped after 150 epochs!')
        break
            
np.save(f'{save_dir}/cluster_labels/cluster_label_{runs[j]}_Durr_DI_AE_{str(n_train_samples)}_minClassic_ALLfeatures.npy', cluster_label)
np.save(f'{save_dir}/centroids/centroids_{runs[j]}_Durr_DI_AE_{str(n_train_samples)}_minClassic_ALLfeatures.npy', centroids)

print('Centroids and labels saved!')