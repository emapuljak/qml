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
import scripts.kmedians as KMed
import scripts.oracle as o
import utils as u
import plots as p

qibo.set_device("/GPU:0")

seed=1234
train_size=[10, 6000]
#run = '01032022_1' #24 latent dim
latent_dim = 8
k=2
save_dir='/eos/user/e/epuljak/private/epuljak/PhD/TN/QIBO/search_algorithms/notebooks/results_kmedians/diJet'

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
    
    # TRAIN K-MEDIANS
    kmedians = KMed.Kmedians(k=k)
    kmedians.fit(inputs)

    loss = kmedians.loss
    centroids_c = kmedians.medians
            
    #np.save(f'{save_dir}/cluster_labels/cluster_label_lat{latent_dim}_{str(train_size[j])}.npy', cluster_label)
    np.save(f'{save_dir}/centroids/final/centroids_lat{latent_dim}_{str(train_size[j])}.npy', centroids_c)
    np.save(f'{save_dir}/loss/final/LOSS_lat{latent_dim}_{str(train_size[j])}.npy', loss)
    print('Centroids and labels saved!')

    
