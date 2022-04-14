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

latent_dims=['6']
runs = ['24032022_1']
n_train_samples=500
save_dir='/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians'

for j in range(len(runs)):
    print(f'========== LATENT DIM {latent_dims[j]} ==========')
    
    # read QCD predicted data (test - SIDE)
    read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/{runs[j]}/'
    file_name = 'latentrep_QCD_sig.h5'
    with h5py.File(read_dir+file_name, 'r') as file:
        data_train = file['latent_space'][:n_train_samples]
    
    # TRAIN Q-MEDIANS
    k = 2 # number of clusters
    #centroids = qkmed.initialize_centroids(data_train, k)   # Intialize centroids
    centroids = np.load('/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/centroids/centroids_random_start_24032022_1_Durr_DI_AE_500.npy')
    tolerance=1e-3

    # run k-medians algorithm
    i = 0
    loss=[]
    cluster_label = np.ones(data_train.shape[0])
    while True:
        new_cluster_label, _ = qkmed.find_nearest_neighbour_DI(data_train,centroids)       # find nearest centers
        print(f'Found cluster assignments for iteration: {i+1}')
        new_centroids = qkmed.find_centroids_GM(data_train,new_cluster_label, clusters=k)               # find centroids
        
        #loss_epoch = np.linalg.norm(centroids - new_centroids)
        loss_epoch = np.linalg.norm(cluster_label - new_cluster_label)
        loss.append(loss_epoch)
        
        # if loss_epoch < tolerance:
        #     centroids = new_centroids
        #     print(f"Converged after {i+1} iterations.")
        #     break
        
        if np.sum(cluster_label != new_cluster_label) == 0: ## no changes in cluster labels
            centroids = new_centroids
            print(f"Converged after {i+1} iterations.")
            break
        print(f'Iteration: {i+1}')
        i += 1     
        centroids = new_centroids
        cluster_label = new_cluster_label
        if i == 150: 
            print('QKmedians stopped after 150 epochs!')
            break
            
    np.save(f'{save_dir}/cluster_labels/cluster_label_{runs[j]}_Durr_DI_AE_{str(n_train_samples)}_minClassic_2_sameRandomStart_testing.npy', cluster_label)
    np.save(f'{save_dir}/centroids/centroids_{runs[j]}_Durr_DI_AE_{str(n_train_samples)}_minClassic_2_sameRandomStart_testing.npy', centroids)
    np.save(f'{save_dir}/LOSS_{runs[j]}_Durr_DI_AE_{str(n_train_samples)}_minClassic_2_sameRandomStart_testing.npy', loss)
    print('Centroids and labels saved!')