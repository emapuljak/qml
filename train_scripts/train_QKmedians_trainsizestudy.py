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

seed=1234
train_size=[10, 6000]
#run = '01032022_1' #24 latent dim
latent_dim = 8
save_dir='/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/corrected_cuts/diJet'

# # read QCD predicted data (test - SIG)
# read_dir =f'/eos/user/e/epuljak/private/epuljak/PhD/Autoencoders/inference_ntb/results/corrected_cuts/{run}/'
# file_name = 'latentrep_QCD_sig.h5'
# with h5py.File(read_dir+file_name, 'r') as file:
#     data_train = file['latent_space'][:]
# read QCD predicted data (test - SIG)
read_dir =f'/eos/user/e/epuljak/private/epuljak/public/AE_data/latent/lat{latent_dim}/'
file_name = 'latentrep_QCD_sig.h5'
with h5py.File(read_dir+file_name, 'r') as file:
    data = file['latent_space']
    l1 = data[:,0,:]
    l2 = data[:,1,:]
    
for j in range(len(train_size)):
    print(f'========== TRAIN SIZE {str(train_size[j])} ==========')
    
    #inputs = data_train[:train_size[j]]
    inputs = np.vstack([l1[:train_size[j]], l2[:train_size[j]]])
    np.random.seed(seed)
    np.random.shuffle(inputs)
    # TRAIN Q-MEDIANS
    k = 2 # number of clusters
    centroids = qkmed.initialize_centroids(inputs, k)   # Intialize centroids
    #centroids = np.load('/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_qmedians/centroids/centroids_random_start_24032022_1_Durr_DI_AE_500.npy')
    tolerance=1.e-3

    # run k-medians algorithm
    i = 0; new_tol=1
    loss=[]
    #cluster_label = np.ones(inputs.shape[0])
    while True:
        #np.save(f'{save_dir}/centroids/final/check_loss/centroids_lat{latent_dim}_{str(train_size[j])}_i{i}.npy', centroids)
        cluster_label, _ = qkmed.find_nearest_neighbour_DI(inputs,centroids)       # find nearest centers
        print(f'Found cluster assignments for iteration: {i+1}')
        new_centroids = qkmed.find_centroids_GM(inputs, cluster_label, centroids, clusters=k)               # find centroids
        # new_centroids = np.zeros([k, data_train.shape[1]])
        # for kk in range(k):
        #     print(f'Searching centroids for cluster {kk}')
        #     points_class_i = data_train[cluster_label==kk]
            # new_centroids[kk,:] = np.median(points_class_i, axis=0)
        print(f'Found new centroids for iteration: {i+1}')
        loss_epoch = np.linalg.norm(centroids - new_centroids)
        #loss_epoch = np.linalg.norm(cluster_label - new_cluster_label)
        loss.append(loss_epoch)
        
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
        # if i == 100:
        #     print('QKmedians stopped after 400 epochs!')
        #     centroids = new_centroids
        #     #cluster_label = new_cluster_label
        #     break
        i += 1     
        centroids = new_centroids
        #cluster_label = new_cluster_label
            
    #np.save(f'{save_dir}/cluster_labels/final/cluster_label_lat{latent_dim}_{str(train_size[j])}_k{k}_npmed.npy', cluster_label)
    np.save(f'{save_dir}/centroids/final/centroids_lat{latent_dim}_{str(train_size[j])}_k{k}_new.npy', centroids)
    np.save(f'{save_dir}/loss/final/LOSS_lat{latent_dim}_{str(train_size[j])}_k{k}_new.npy', loss)
    print('Centroids and labels saved!')

    
