import numpy as np
import random
import os
seed=1234
# os.environ['PYTHONHASHSEED'] = str(seed)
# random.seed(seed)
# np.random.seed(seed)
import sys
sys.path.append('../')
sys.path.append('../../')
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

qibo.set_device("/GPU:1")

#runs=[]
latent_dims=['4', '16', '24', '32']
#runs = ['29032022_1', ]
#latent_dims=['40']
#runs=['14032022_1']
n_train_samples=600
#n_start=2000
#x=4
save_dir='/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts/diJet'

for j in range(len(latent_dims)):
    print(f'========== LATENT DIM {latent_dims[j]} ==========')
    
    # read QCD predicted data (test - SIDE)
    read_dir =f'/eos/user/e/epuljak/private/epuljak/public/AE_data/latent/lat{latent_dims[j]}/'
    #read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/corrected_cuts/{runs[j]}/'
    file_name = 'latentrep_QCD_sig.h5'
    with h5py.File(read_dir+file_name, 'r') as file:
        data = file['latent_space']
        l1 = data[:,0,:]
        l2 = data[:,1,:]
        
        #r_index = np.random.choice(list(range(l1.shape[0])), size=int(n_train_samples))
        data_train = np.vstack([l1[:n_train_samples], l2[:n_train_samples]])
        np.random.seed(seed)
        np.random.shuffle(data_train)
    
    # TRAIN Q-MEDIANS
    k = 2 # number of clusters
    centroids = qkmed.initialize_centroids(data_train, k)   # Intialize centroids
    #centroids = np.load('/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/centroids/centroids_random_start_24032022_1_Durr_DI_AE_500.npy')
    tolerance=1.e-3

    # run k-medians algorithm
    i = 0; new_tol=1
    loss=[]
    #cluster_label = np.ones(data_train.shape[0])
    while True:
        cluster_label, _ = qkmed.find_nearest_neighbour_DI(data_train,centroids)       # find nearest centers
        print(f'Found cluster assignments for iteration: {i+1}')
        new_centroids = qkmed.find_centroids_GM(data_train, cluster_label, centroids, clusters=k)               # find centroids
        # new_centroids = np.zeros([k, data_train.shape[1]])
        # for kk in range(k):
        #     print(f'Searching centroids for cluster {kk}')
        #     points_class_i = data_train[cluster_label==kk]
            # new_centroids[kk,:] = np.median(points_class_i, axis=0)
        print(f'Found new centroids for iteration: {i+1}')
        loss_epoch = np.linalg.norm(centroids - new_centroids)
        #loss_epoch = np.linalg.norm(cluster_label - new_cluster_label)
        loss.append(loss_epoch)
        print(loss_epoch)
        if loss_epoch < tolerance:
            centroids = new_centroids
            print(f"Converged after {i+1} iterations.")
            break
        elif loss_epoch > tolerance and i > new_tol*100:     # if after 200*new_tol epochs, difference != 0, lower the tolerance
            print(f'Rising the tolerance after {i}th iteration')
            #if new_tol>1: tolerance *= 2
            tolerance *= 2
            print(f'New tolerance: {tolerance}')
            new_tol += 1
        
        # if np.sum(cluster_label != new_cluster_label) == 0: ## no changes in cluster labels
        #     centroids = new_centroids
        #     print(f"Converged after {i+1} iterations.")
        #     break
        print(f'Iteration: {i+1}')
        # if i == 250: 
        #     print('QKmedians stopped after 250 epochs!')
        #     centroids = new_centroids
        #     #cluster_label = new_cluster_label
        #     break
        i += 1
        centroids = new_centroids
        #cluster_label = new_cluster_label
        #np.save(f'{save_dir}/centroids/final/centroids_lat{latent_dims[j]}_{str(n_train_samples)}_iter{i}.npy', centroids)
    np.save(f'{save_dir}/centroids/final/centroids_lat{latent_dims[j]}_{str(n_train_samples)}_k{k}_new.npy', centroids)
    np.save(f'{save_dir}/loss/final/LOSS_lat{latent_dims[j]}_{str(n_train_samples)}_k{k}_new.npy', loss)
    print('Centroids and loss saved!')
