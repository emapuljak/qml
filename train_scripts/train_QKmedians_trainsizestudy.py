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

train_size=['2000', '1000', '500', '300', '100', '50', '30']
run = '01032022_2' #32
save_dir='/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians'

# read QCD predicted data (test - SIDE)
read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/{run}/'
file_name = 'latentrep_QCD_sig.h5'
with h5py.File(read_dir+file_name, 'r') as file:
    data_train = file['latent_space'][:]
        
for i in range(len(train_size)):
    print(f'========== TRAIN SIZE {train_size[i]} ==========')
    
    inputs = data_train[:n_train_samples]
    
    # TRAIN Q-MEDIANS
    k = 2 # number of clusters
    centroids = qkmed.initialize_centroids(inputs, k)   # Intialize centroids
    tolerance=1e-3

    # run k-medians algorithm
    i = 0
    while True:
        cluster_label, _ = qkmed.find_nearest_neighbour_DI(inputs,centroids)       # find nearest centers
        print(f'Found cluster assignments for iteration: {i+1}')
        new_centroids = qkmed.find_centroids_GM(inputs,cluster_label, clusters=k)               # find centroids

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
            
    np.save(f'{save_dir}/cluster_labels/cluster_label_{run}_Durr_DI_AE_{str(train_size[i])}.npy', cluster_label)
    np.save(f'{save_dir}/centroids/centroids_{run}_Durr_DI_AE_{str(train_size[i])}.npy', centroids)

    print('Centroids and labels saved!')