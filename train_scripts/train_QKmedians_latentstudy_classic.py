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
import scripts.kmedians as KMed
import scripts.oracle as o
import utils as u
import plots as p

qibo.set_device("/GPU:0")

#latent_dims=['4', '8', '16', '24', '32', '40']
#runs=['lat8_final_withinit_1', 'lat8_final_withinit_2', 'lat8_final_withoutinit_1', 'lat8_final_withoutinit_2']
latent_dims=['8']
#runs = ['29032022_1', ]
#latent_dims=['40']
#runs=['14032022_1']
n_train_samples=600
#n_start=2000
#x=4
k=15
save_dir='/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_kmedians/diJet'

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
        
        #r_index = np.random.choice(list(range(l1.shape[0])), size=int(n_train_samples)) # take first X train 
        #data_train = np.vstack([l1[:n_train_samples], l2[:n_train_samples]])
        #np.random.shuffle(data_train) # set a seed??? - set global numpy seed
        
        #r_index = np.random.choice(list(range(l1.shape[0])), size=int(n_train_samples))
        data_train = np.vstack([l1[:n_train_samples], l2[:n_train_samples]])
        np.random.seed(seed)
        np.random.shuffle(data_train)
    
    # TRAIN K-MEDIANS
    kmedians = KMed.Kmedians(k=k)
    kmedians.fit(data_train)

    loss = kmedians.loss
    centroids_c = kmedians.centroids
            
    #np.save(f'{save_dir}/cluster_labels/cluster_label_lat{latent_dims[j]}_{str(n_train_samples)}.npy', cluster_label)
    np.save(f'{save_dir}/centroids/final/centroids_lat{latent_dims[j]}_{str(n_train_samples)}_k{k}.npy', centroids_c)
    np.save(f'{save_dir}/loss/final/LOSS_lat{latent_dims[j]}_{str(n_train_samples)}_k{k}.npy', loss)
    print('Centroids and labels saved!')
